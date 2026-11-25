#!/usr/bin/env python3
"""
InferPlan cost model: learn from measured runs, predict the rest.

This module never starts a server and never measures anything. It reads the
runs that runner.py already recorded and fits a small, physics-shaped model
so the planner can price configurations that were never run.

The model is deliberately NOT a black box. It is a roofline estimate with
three learned corrections, each of which came out of a measured finding:

  1. decode correction   - measured step time over the roofline step time.
                           Finding #6: 6.9 ms measured vs 5.2 ms predicted.
  2. prefill time as a power law in the prompt tokens the cache did NOT
                           remove. Measured on this data: TTFT grows as
                           tokens^0.60, i.e. SUB-linearly, because a longer
                           prefill keeps the GPU busier per kernel launch.
                           A straight line gets 30% error here; the power law
                           gets 4%.
  3. queue penalty       - when client concurrency exceeds the server's
                           max-num-seqs, later requests wait for a whole
                           batch. Finding #7: TTFT jumped 28x at that point.

Every prediction carries an error bar, because Finding #15 showed the noise
floor moves between sessions (0.5% to 6.2%), so a point estimate alone would
overstate what is known.

Usage
    python costmodel.py --ledger ledger.jsonl            # fit and validate
    python costmodel.py --ledger ledger.jsonl --predict \
        --quantization awq --max-num-seqs 16 --shared-prefix-tokens 1024
"""
import argparse
import json
import math
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ----------------------------------------------------------------------------
# hardware and model facts
#
# These are inputs, not learned values. The planner passes them in; the table
# is only a convenience so the common cases need no arguments.
# ----------------------------------------------------------------------------
GPU_BANDWIDTH_GBPS = {
    "RTX 3050 Laptop GPU": 192.0,
    "RTX 3050": 192.0,
    "RTX 3060 Laptop GPU": 336.0,
    "RTX 4090": 1008.0,
    "A10G": 600.0,
    "A100-SXM4-40GB": 1555.0,
    "H100": 3350.0,
    "T4": 320.0,
}

MODEL_PARAMS = {
    "Qwen/Qwen2.5-0.5B-Instruct": 494_000_000,
    "Qwen/Qwen2.5-1.5B-Instruct": 1_540_000_000,
    "Qwen/Qwen2.5-1.5B-Instruct-AWQ": 1_540_000_000,
    "microsoft/Phi-3-mini-4k-instruct": 3_820_000_000,
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 1_100_000_000,
}

BYTES_PER_PARAM = {
    None: 2.0, "none": 2.0, "fp16": 2.0, "half": 2.0, "bfloat16": 2.0,
    "fp8": 1.0, "int8": 1.0,
    "awq": 0.5, "gptq": 0.5, "int4": 0.5,
}


def weight_bytes(model: str, quantization: Optional[str],
                 params: Optional[int] = None) -> float:
    n = params or MODEL_PARAMS.get(model)
    if n is None:
        raise ValueError(f"unknown parameter count for {model!r}; pass params=")
    return n * BYTES_PER_PARAM.get(quantization, 2.0)


def roofline_step_ms(model: str, quantization: Optional[str],
                     bandwidth_gbps: float, params: Optional[int] = None) -> float:
    """
    Lower bound on the time for one decode step: every weight is read once.

    This is the scale-free part of the model. It holds for a 0.5B model on a
    laptop GPU and for a 100B model on a cluster; only the numbers change.
    """
    gb = weight_bytes(model, quantization, params) / 1e9
    return gb / bandwidth_gbps * 1000.0


# ----------------------------------------------------------------------------
# the runs we learn from
# ----------------------------------------------------------------------------
@dataclass
class Run:
    label: str
    config: dict
    metrics: dict
    outcome: str
    noise_pct: float
    # Row-level facts that are not configuration: when it ran, and whether a
    # reference control vouched for the machine state at the time. Findings
    # #31-#33: without that vouch a run can be 6x off and look perfect.
    meta: dict = field(default_factory=dict)

    @property
    def controlled(self) -> bool:
        """Was a reference control taken around this run, and did it hold?"""
        if self.meta.get("uncontrolled"):
            return False
        if self.meta.get("contaminated"):
            return False
        return self.meta.get("reference_tok_s") is not None

    def g(self, key, default=None):
        v = self.config.get(key, default)
        return default if v is None else v

    def m(self, key) -> Optional[float]:
        v = self.metrics.get(key)
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if f != f else f          # drop NaN

    # ---- derived features -------------------------------------------------
    @property
    def cached_prefix_tokens(self) -> float:
        """
        Tokens of prefill that a warm cache removes.

        Prefix caching only removes work that REPEATS. With no shared prefix
        there is nothing to remove, which is why the sp0 pair in Finding #18
        showed almost no difference. The chat template's own header does
        repeat, and that is the small residue seen there.
        """
        if not self.g("prefix_caching", True):
            return 0.0
        return float(self.g("shared_prefix_tokens", 0))

    @property
    def full_price_tokens(self) -> float:
        """Prompt tokens that must be prefilled from scratch."""
        base = float(self.g("base_prompt_tokens", 38))   # the rotating prompts
        shared = float(self.g("shared_prefix_tokens", 0))
        return base + shared - self.cached_prefix_tokens

    @property
    def prefill_tokens(self) -> float:
        """Kept as an alias; the planner reports this number."""
        return self.full_price_tokens

    @property
    def waves(self) -> int:
        """
        How many server-side batches the client's load is split into.

        Finding #7: with concurrency 16 against max-num-seqs 8 the second
        half of a BURST waits for the first half to finish decoding.
        """
        c = int(self.g("concurrency", 8))
        m = int(self.g("max_num_seqs", 8))
        return max(1, math.ceil(c / max(m, 1)))

    @property
    def oversubscription(self) -> float:
        """
        How far the client's concurrency exceeds the server's batch slots.

        Finding #22: the knee sits exactly at concurrency == slots, on every
        slot count tested. Past it, throughput gains 3-7% and p99 doubles.
        """
        c = float(self.g("concurrency", 8))
        m = float(self.g("max_num_seqs", 8))
        return max(0.0, c / max(m, 1.0) - 1.0)

    # A "burstiness" factor used to live here. It was added to explain a run
    # that measured 1076 ms against a 1984 ms prediction, and it was WRONG:
    # that run had been measured next to an orphan server. The clean rerun of
    # the same configuration measures 1972 ms, which the plain full-wave model
    # predicts to within 1%. The extra term was fitted to an artifact, and it
    # is gone. Little's Law gives the whole thing:
    #
    #     in flight = c,  throughput R = slots / (out_tokens * step)
    #     latency   L = c / R = out * step * c / slots
    #     service   S = out * step
    #     wait      = L - S = batch_decode * (c/slots - 1)
    #               = batch_decode * oversubscription
    #
    # Checked against all nine clean knee configs: within 6% on seven of
    # them, and 15-18% low on the two at 32 slots, where the power cap of
    # Finding #21 slows the batch down.


def audit(runs: list, max_noise_pct: float = 25.0) -> dict:
    """
    Check the ledger for runs that are not measurements, BEFORE fitting.

    A model will cheerfully absorb a broken run into a parameter and report a
    plausible-looking number. On 23 Nov a contaminated knee sweep pushed the
    queue coefficient to 8.06, which would mean a request waits for eight full
    batch decodes: arithmetically fine, physically nonsense. The fix is not a
    better fit, it is to notice the data first.

    Two checks, both cheap:
      1. internal noise  - a config whose own passes disagree by more than
                           max_noise_pct did not measure one thing.
      2. monotonicity    - more batch slots must not yield less throughput at
                           the same client concurrency. If it does, something
                           outside the configuration was moving.
    """
    noisy = [r for r in runs if r.noise_pct > max_noise_pct]

    # Monotonicity in server batch slots, holding EVERYTHING ELSE fixed.
    #
    # The comparison only means something between runs that differ in the slot
    # count alone. An earlier version grouped by concurrency only, so the
    # prefix-curve runs (all at 8 slots and concurrency 8, differing in shared
    # prefix length) were compared against each other and their perfectly
    # ordinary throughput differences were reported as inversions. An audit
    # that cries wolf gets ignored, which defeats its purpose.
    groups = {}
    for r in runs:
        if r.outcome not in ("OK", "PARTIAL") or not r.m("throughput_tok_s"):
            continue
        signature = (
            r.g("model"), r.g("quantization"), r.g("dtype"),
            r.g("max_model_len"), r.g("kv_cache_dtype"),
            r.g("shared_prefix_tokens"), r.g("prefix_caching"),
            r.g("chunked_prefill"), r.g("max_tokens"),
            int(r.g("concurrency", 8)),
        )
        groups.setdefault(signature, []).append(r)

    inversions = []
    for signature, rs in groups.items():
        # One run per slot count; a repeat of the same slot count belongs to
        # the drift check below, not here.
        by_slots = {}
        for r in rs:
            by_slots.setdefault(int(r.g("max_num_seqs", 8)), []).append(r)
        if len(by_slots) < 2:
            continue
        ordered = sorted((slots, max(g, key=lambda r: r.m("throughput_tok_s")))
                         for slots, g in by_slots.items())
        for (sa, a), (sb, b) in zip(ordered, ordered[1:]):
            ta, tb = a.m("throughput_tok_s"), b.m("throughput_tok_s")
            if ta and tb and tb < ta * 0.9:      # 10% tolerance for noise
                inversions.append({
                    "concurrency": signature[-1],
                    "fewer_slots": sa, "tput_a": ta,
                    "more_slots": sb, "tput_b": tb,
                    "labels": (a.label, b.label)})

    # Same config measured more than once across sessions: how far apart?
    drift = []
    seen = {}
    for r in runs:
        if r.outcome not in ("OK", "PARTIAL") or not r.m("throughput_tok_s"):
            continue
        key = (r.g("model"), r.g("quantization"), r.g("dtype"),
               r.g("max_num_seqs"), r.g("concurrency"),
               r.g("shared_prefix_tokens"), r.g("prefix_caching"),
               r.g("max_model_len"))
        seen.setdefault(key, []).append(r)
    for key, rs in seen.items():
        if len(rs) < 2:
            continue
        ts = [r.m("throughput_tok_s") for r in rs]
        lo, hi = min(ts), max(ts)
        if hi > lo * 1.1:
            drift.append({"labels": [r.label for r in rs],
                          "low": lo, "high": hi,
                          "spread_pct": (hi / lo - 1) * 100})

    return {"noisy": noisy, "inversions": inversions, "drift": drift}


def fit_two_term(obs: list):
    """
    Fit y = a*x1 + b*x2 with no intercept, by least squares on 3x3 normal
    equations. Returns (a, b, mape) or None.

    No intercept on purpose: the inputs are already residuals above a
    baseline, so a constant term here would silently refit the part the
    caller has already explained.

    Refuses when the two predictors are collinear across the data, which is
    exactly what happens if every run has caching off - then full-price and
    context tokens are the same number and no fit can tell their
    coefficients apart. The prefix sweeps measure both settings at each
    length for this reason.
    """
    if len(obs) < 3:
        return None
    s11 = sum(x1 * x1 for x1, _, _ in obs)
    s22 = sum(x2 * x2 for _, x2, _ in obs)
    s12 = sum(x1 * x2 for x1, x2, _ in obs)
    s1y = sum(x1 * y for x1, _, y in obs)
    s2y = sum(x2 * y for _, x2, y in obs)
    den = s11 * s22 - s12 * s12
    if abs(den) < 1e-12 or s11 <= 0 or s22 <= 0:
        return None
    # Collinearity check: if the two columns are nearly proportional the
    # split between them is arbitrary, however good the overall fit looks.
    corr = s12 / ((s11 ** 0.5) * (s22 ** 0.5))
    if abs(corr) > 0.995:
        return None
    a = (s1y * s22 - s2y * s12) / den
    b = (s2y * s11 - s1y * s12) / den
    if a < 0 or b < 0:
        # A negative coefficient would mean a longer prompt makes decoding
        # faster. More likely the residuals are noise.
        return None
    errs = []
    for x1, x2, y in obs:
        if abs(y) < 1e-9:
            continue
        errs.append(abs((a * x1 + b * x2) - y) / abs(y))
    err = sum(errs) / len(errs) * 100 if errs else 0.0
    return a, b, err


def fit_power_law(obs: list, floor_hint: float = 0.0,
                  alpha_lo: float = 1.0, alpha_hi: float = 2.0):
    """
    Fit y = c0 + c1 * x^alpha by least squares, with alpha found by a coarse
    search. Returns (c0, c1, alpha, mape) or None.

    Refuses rather than guesses when there are fewer than three distinct x
    values: two points fit any two-parameter curve exactly and say nothing
    about its shape, which is the same reason the saturation fit refuses
    below three slot counts.
    """
    xs_all = sorted({x for x, _ in obs})
    if len(xs_all) < 3:
        return None
    best = None
    alpha = alpha_lo
    while alpha <= alpha_hi + 1e-9:
        xs = [x ** alpha for x, _ in obs]
        ys = [y for _, y in obs]
        n = len(xs)
        sx, sy = sum(xs), sum(ys)
        sxx = sum(v * v for v in xs)
        sxy = sum(v * y for v, y in zip(xs, ys))
        den = n * sxx - sx * sx
        this_alpha = alpha
        alpha += 0.01
        if abs(den) < 1e-12:
            continue
        c1 = (n * sxy - sx * sy) / den
        c0 = (sy - c1 * sx) / n
        # A negative coefficient would mean a bigger batch steps FASTER, and a
        # negative intercept that an empty batch takes negative time.
        if c1 < 0 or c0 < 0:
            continue
        err = sum(abs((c0 + c1 * v) - y) / y for v, y in zip(xs, ys)) / n * 100
        if best is None or err < best[3]:
            best = (c0, c1, this_alpha, err)
    if best and floor_hint and best[0] < floor_hint * 0.5:
        # The intercept is the cost of reading the weights once, so it cannot
        # sit far below the roofline step. If it does, the fit has absorbed
        # the floor into the batch term and will mispredict small batches.
        return None
    return best


def load_runs(path: Path) -> list:
    """Read runner.py's ledger. Crashed runs are kept but flagged."""
    runs = []
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run a sweep first")
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("kind") == "reference":
            # The control's own readings are machine-state records, not
            # configurations to recommend. They are read by the drift check,
            # not by the search.
            continue
        cfg = row.get("effective_config") or row.get("config") or {}
        runs.append(Run(label=row.get("label", "?"),
                        config=cfg,
                        metrics=row.get("median") or {},
                        outcome=row.get("outcome", "?"),
                        noise_pct=float(row.get("noise_pct") or 0.0),
                        meta={k: row.get(k) for k in
                              ("timestamp", "sweep", "reference_tok_s",
                               "reference_baseline_tok_s",
                               "reference_drift_pct", "contaminated")}))
    # Later runs of the same label supersede earlier ones.
    latest = {}
    for r in runs:
        latest[r.label] = r
    return list(latest.values())


# Keys bench.py knows about, under the names the Run accessors expect. It is
# a CLIENT, so it knows what it asked for and nothing about how the server
# was started: quantization, dtype, max_model_len and prefix_caching are
# genuinely unknown here and are deliberately left absent rather than
# guessed. Absent fields keep such a run in a workload shape of its own,
# which is the honest outcome - but the run still counts as an observation.
_BENCH_ARG_MAP = {
    "model": "model",
    "concurrency": "concurrency",
    "server_max_num_seqs": "max_num_seqs",
    "max_tokens": "max_tokens",
    "shared_prefix_tokens": "shared_prefix_tokens",
}


def load_bench_runs(json_dir: Path) -> list:
    """
    Read the per-run JSON that bench.py writes, for runs started by hand.

    Finding #30: a 128-slot measurement that contradicted the saturation fit
    was invisible to the planner for three days, because the planner read
    only runner.py's ledger and the run had been launched through bench.py.
    Fourteen JSON records sat in runs/ that nothing ever opened. A measurement
    that cannot reach the model cannot correct it, so they are read now.
    """
    runs = []
    if not json_dir or not json_dir.exists():
        return runs
    for f in sorted(json_dir.glob("*.json")):
        try:
            row = json.loads(f.read_text(errors="replace"))
        except (json.JSONDecodeError, OSError):
            continue
        a = row.get("args") or {}
        if not a.get("label"):
            continue
        cfg = {}
        for src, dst in _BENCH_ARG_MAP.items():
            v = a.get(src)
            if v is not None and v != "":
                cfg[dst] = v
        if "max_num_seqs" not in cfg and cfg.get("concurrency"):
            # bench.py allows --server-max-num-seqs to be omitted. Treating
            # the client concurrency as the slot count would invent a
            # measurement; recording nothing is correct, so the run is kept
            # out of the slot analysis entirely.
            cfg.pop("concurrency", None)
        cfg["source"] = "bench"
        # bench.py has no reference control, so nothing vouches for the
        # machine state these were taken in. Finding #33: three such runs in
        # runs/ read 801, 1213 and 3083 tok/s for configurations that a
        # controlled sweep measured at 4714 and 5339. They are kept as a
        # record and used to raise a question, never to set a ceiling.
        runs.append(Run(label=a["label"], config=cfg,
                        metrics=row.get("median") or {},
                        outcome=row.get("status", "?"),
                        noise_pct=float(row.get("noise_pct") or 0.0),
                        meta={"uncontrolled": True, "source_file": f.name}))
    latest = {}
    for r in runs:
        latest[r.label] = r
    return list(latest.values())


def load_all_runs(ledger: Path, json_dir: Optional[Path] = None) -> list:
    """
    The ledger plus any hand-run benchmarks it does not contain.

    The ledger wins on a label collision: runner.py records the full server
    configuration, bench.py cannot.
    """
    runs = load_runs(ledger)
    have = {r.label for r in runs}
    extra = [r for r in load_bench_runs(json_dir) if r.label not in have]
    return runs + extra


# ----------------------------------------------------------------------------
# small least squares, so the module needs no numpy
# ----------------------------------------------------------------------------
def solve(a: list, b: list) -> list:
    """Gaussian elimination with partial pivoting on a small square system."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise ValueError("singular system: features are collinear")
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def least_squares(rows: list, targets: list) -> list:
    """Ordinary least squares via the normal equations."""
    k = len(rows[0])
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    atb = [sum(r[i] * t for r, t in zip(rows, targets)) for i in range(k)]
    return solve(ata, atb)


def mape(actual: list, predicted: list) -> float:
    pairs = [(a, p) for a, p in zip(actual, predicted) if a]
    if not pairs:
        return float("nan")
    return statistics.mean(abs(p - a) / abs(a) for a, p in pairs) * 100


# ----------------------------------------------------------------------------
# the fitted model
# ----------------------------------------------------------------------------
@dataclass
class CostModel:
    gpu: str
    bandwidth_gbps: float
    gpu_cost_per_hour: float = 0.35

    # learned
    decode_correction: dict = field(default_factory=dict)   # key -> factor
    # key -> (intercept_ms, coef, alpha). The decode STEP as a function of how
    # many sequences actually share it. Finding #37: the single correction
    # factor above is a per-step figure used as a per-request one, which is
    # the same thing only at a batch of one. It predicted 1113 ms end to end
    # for 64, 96 and 128 slots alike, against 1641, 2319 and 2972 measured.
    decode_step: dict = field(default_factory=dict)
    # key -> (ms_per_full_price_token, ms_per_context_token,
    #         baseline_full, baseline_context). Finding #39.
    decode_prompt: dict = field(default_factory=dict)
    decode_prompt_suspect: bool = False
    fitted_on_controlled: bool = False
    # ttft = intercept + coef * full_price^alpha + cached_ms_per_token * cached
    prefill_coef: float = 0.0
    prefill_alpha: float = 1.0
    prefill_intercept_ms: float = 0.0
    cached_ms_per_token: float = 0.0
    prefill_form: str = "none"
    queue_k: float = 1.0        # learned; 1.0 = a full wave of waiting
    queue_suspect: bool = False  # set when the fitted k was physically absurd
    noise_floor_pct: float = 0.0
    n_runs: int = 0
    fit_report: list = field(default_factory=list)

    # ---- fitting ----------------------------------------------------------
    def fit(self, runs: list, max_noise_pct: float = 25.0):
        # A run whose own passes disagree wildly is not an observation of one
        # configuration, so it is dropped rather than averaged in.
        dropped = [r for r in runs if r.noise_pct > max_noise_pct]
        if dropped:
            self.fit_report.append(
                f"dropped {len(dropped)} run(s) with internal noise above "
                f"{max_noise_pct:.0f}%: " +
                ", ".join(f"{r.label[:28]} ({r.noise_pct:.0f}%)"
                          for r in dropped[:3]))
        runs = [r for r in runs if r.noise_pct <= max_noise_pct]

        usable = [r for r in runs
                  if r.outcome in ("OK", "PARTIAL") and r.m("tpot_p50")]
        if len(usable) < 2:
            raise ValueError(f"need at least 2 successful runs, got {len(usable)}")

        # Prefer runs a reference control vouched for, exactly as the
        # saturation fit does. Findings #31-#35: this machine moved 6x on one
        # configuration within a day, so fitting across sittings makes every
        # coefficient an average over machine states nobody recorded. The
        # decode-step law came out at 12% MAPE on the mixed ledger against
        # 2.1% on the controlled runs alone, with alpha pulled from 1.42 to
        # 1.87 - a physical exponent bent to absorb platform drift.
        vouched = [r for r in usable if r.controlled]
        if len(vouched) >= 8:
            left_out = len(usable) - len(vouched)
            if left_out:
                self.fit_report.append(
                    f"using {len(vouched)} reference-controlled run(s); "
                    f"{left_out} older run(s) with no control are excluded "
                    f"(finding #35)")
            else:
                # Said out loud even when nothing was excluded. Silence here
                # reads as "the check did not run", and that ambiguity cost
                # an hour of looking for a bug that was not there.
                self.fit_report.append(
                    f"all {len(vouched)} run(s) were bracketed by a reference "
                    f"control, so the coefficients below rest on one "
                    f"vouched-for machine state")
            usable = vouched
            self.fitted_on_controlled = True
        elif len(vouched):
            self.fit_report.append(
                f"only {len(vouched)} controlled run(s), fewer than the 8 "
                f"needed to fit on them alone, so ALL {len(usable)} runs are "
                f"used and every coefficient carries an unmeasured "
                f"machine-state term")
        else:
            self.fit_report.append(
                "NO run in this ledger was bracketed by a reference control, "
                "so nothing vouches for the machine states these "
                "coefficients average over. Re-run sweeps with "
                "--reference-every to fix this.")
        self.n_runs = len(usable)

        # 1. decode correction, one per (model, quantization, dtype)
        groups = {}
        for r in usable:
            key = (r.g("model"), r.g("quantization"), r.g("dtype"))
            groups.setdefault(key, []).append(r)

        for key, rs in groups.items():
            model, quant, _ = key
            try:
                theory = roofline_step_ms(model, quant, self.bandwidth_gbps)
            except ValueError as e:
                self.fit_report.append(f"skipped {key}: {e}")
                continue
            measured = statistics.median(r.m("tpot_p50") for r in rs)
            self.decode_correction[key] = measured / theory
            self.fit_report.append(
                f"decode  {model.split('/')[-1]:<28} {str(quant):<6} "
                f"theory {theory:5.2f} ms  measured {measured:5.2f} ms  "
                f"correction {measured / theory:4.2f}  ({len(rs)} runs)")

            # 1b. the step as a function of the batch that shares it.
            #
            # The batch is min(concurrency, max_num_seqs), not max_num_seqs:
            # an under-driven server decodes only what arrived. The data says
            # so plainly - b16_c8 stepped in 7.96 ms and b8_c8 in 7.89, while
            # b16_c16 took 8.55. The slot limit is a cap, the concurrency is
            # the load, and the smaller of the two is the batch.
            #
            # The form is a power law with a floor, fitted the same way as
            # prefill below:
            #
            #     step_ms = c0 + c1 * batch^alpha
            #
            # Measured here: c0 7.50 ms, c1 0.0169, alpha 1.42, MAPE 2.1%
            # across batches of 4 to 128. c0 lands within 2% of the roofline
            # step times the correction, which is what it should be: the
            # weights are read once per step whatever the batch, and
            # everything above that floor is per-sequence work - KV reads and
            # attention - growing faster than the batch because each sequence
            # also lengthens the attention each other sequence computes.
            # Fitted on SHORT-PROMPT runs only. The prefill fit below uses
            # runs with more than 100 full-price tokens; this one needs the
            # opposite, and for the same reason: each sub-model is fitted
            # where its own effect dominates and the others are negligible.
            #
            # With chunked prefill on (vLLM's default), prefill chunks are
            # interleaved with decode steps, so a long prompt inflates TPOT.
            # At a batch of 8 the measured TPOT ranges from 7.9 ms on
            # 38-token prompts to about 17.7 ms on the 2048-token shared
            # prefix runs. Fitting a batch law across that forces one
            # parameter to carry two effects: the first attempt did exactly
            # that, drove alpha to the 2.0 search boundary, and came out at
            # 15.1% MAPE where the short-prompt runs alone give 2.1%.
            obs = []
            skipped_long = 0
            for r in rs:
                tp = r.m("tpot_p50")
                batch = min(int(r.g("concurrency", 8)), int(r.g("max_num_seqs", 8)))
                if not tp or batch <= 0:
                    continue
                # The filter must use the PROMPT, not the full-price tokens.
                # A cache hit makes full_price_tokens 38 while the prompt is
                # still 2086 long, and chunked prefill interleaves the actual
                # prompt whether or not the cache served it. Filtering on
                # full_price let every cache-hit prefix run back in and held
                # this fit at 12% MAPE with alpha bent to 1.87.
                # prefill_tokens is an alias for full_price_tokens, so it is
                # no help here; the prompt has to be computed.
                prompt = (float(r.g("base_prompt_tokens", 38))
                          + float(r.g("shared_prefix_tokens", 0)))
                if prompt > 100:
                    skipped_long += 1
                    continue
                obs.append((batch, tp))
            step_fit = fit_power_law(obs, floor_hint=theory)

            # 1c. what a long prompt adds to the step, on top of the batch.
            #
            # Fitted on the runs the batch law had to leave out, against the
            # residual the batch law cannot explain. Two predictors, both
            # physical:
            #
            #   full-price prefill tokens - with chunked prefill on (vLLM's
            #       default) prefill chunks are interleaved with decode steps,
            #       so prompt tokens the cache did NOT remove steal decode
            #       time. A cache hit avoids this term entirely.
            #
            #   context tokens - decode attends over the whole sequence, so a
            #       2086-token context costs more per token than a 38-token
            #       one. A cache hit does NOT avoid this: the sequence is as
            #       long as it is either way.
            #
            # Measured here: 0.00305 ms per full-price token and 0.00041 ms
            # per context token, a ratio of about 7:1, with an intercept of
            # 7.48 ms - which is the batch law's own intercept of 7.50 ms,
            # arrived at from different runs. Two fits agreeing on a constant
            # neither was told about is the only confirmation worth having.
            #
            # This is what held TTFT at 21.5% MAPE and put sp2048 at -42%.
            if step_fit:
                # Named step_mape, not mape: a local called `mape` here
                # shadowed the module-level mape() that this same method
                # calls further down, and the fit died with "'float' object
                # is not callable".
                c0, c1, alpha, step_mape = step_fit
                self.decode_step[key] = (c0, c1, alpha)
                self.fit_report.append(
                    f"        step = {c0:.2f} ms + {c1:.4f} * batch^{alpha:.2f}"
                    f"   MAPE {step_mape:.1f}%  ({len(obs)} short-prompt "
                    f"runs, batches "
                    f"{min(b for b, _ in obs)}-{max(b for b, _ in obs)})")
                # Now explain what the batch law could not: fit the
                # residual of the long-prompt runs against the two prompt
                # terms. Fitted on the residual rather than on TPOT itself so
                # the batch term is not refitted and cannot drift.
                base_full = min((r.full_price_tokens for r in rs), default=38.0)
                base_ctx = min(
                    (float(r.g("base_prompt_tokens", 38))
                     + float(r.g("shared_prefix_tokens", 0)) for r in rs),
                    default=38.0)
                pobs = []
                for r in rs:
                    tp = r.m("tpot_p50")
                    bt = min(int(r.g("concurrency", 8)),
                             int(r.g("max_num_seqs", 8)))
                    if not tp or bt <= 0:
                        continue
                    ctx = (float(r.g("base_prompt_tokens", 38))
                           + float(r.g("shared_prefix_tokens", 0)))
                    extra_full = max(r.full_price_tokens - base_full, 0.0)
                    extra_ctx = max(ctx - base_ctx, 0.0)
                    if extra_full <= 0 and extra_ctx <= 0:
                        continue
                    residual = tp - (c0 + c1 * max(bt, 1) ** alpha)
                    pobs.append((extra_full, extra_ctx, residual))
                prompt_fit = fit_two_term(pobs)
                if prompt_fit:
                    pf, pc_, _ = prompt_fit
                    self.decode_prompt[key] = (pf, pc_, base_full, base_ctx)
                    # Scored on TPOT, not on the residual. Residuals sit near
                    # zero, so a percentage error against them is enormous and
                    # meaningless - the first version printed "residual MAPE
                    # 66.3%" for a fit that predicts TPOT to within 4.7%. A
                    # frightening number that measures nothing is its own kind
                    # of dishonesty.
                    errs = []
                    for r2 in rs:
                        tp2 = r2.m("tpot_p50")
                        bt2 = min(int(r2.g("concurrency", 8)),
                                  int(r2.g("max_num_seqs", 8)))
                        if not tp2 or bt2 <= 0:
                            continue
                        ctx2 = (float(r2.g("base_prompt_tokens", 38))
                                + float(r2.g("shared_prefix_tokens", 0)))
                        pred2 = (c0 + c1 * max(bt2, 1) ** alpha
                                 + pf * max(r2.full_price_tokens - base_full, 0.0)
                                 + pc_ * max(ctx2 - base_ctx, 0.0))
                        errs.append(abs(pred2 - tp2) / tp2)
                    tpot_mape = sum(errs) / len(errs) * 100 if errs else 0.0
                    # The coefficient line is written AFTER the paired check
                    # has decided whether the full-price term survives.
                    # Printing it first announced "0.00002 ms per full-price
                    # token" two lines above "the full-price term is
                    # DROPPED", which reads as a measurement followed by an
                    # apology. A report should state what the model ended up
                    # believing, not the order the code worked it out in.
                    report_coefficients = []

                    # PAIRED CHECK. Two runs at the same shared-prefix length,
                    # one with caching and one without, have identical
                    # context and differ only in full-price tokens. Their
                    # measured gap is therefore a direct reading of the
                    # first coefficient, and the fit must reproduce it.
                    #
                    # This exists because a fit with 3.1% MAPE on TPOT put
                    # 0.00002 ms on full-price and 0.00350 on context - the
                    # reverse of the truth. sp2048 measured 8.44 ms cached
                    # and 14.45 ms uncached at the same context, so the
                    # full-price term carries 6 ms that the fit had assigned
                    # elsewhere. A good aggregate score does not make a
                    # model right, and the only way to catch that is a
                    # comparison the aggregate cannot average away.
                    pairs = {}
                    for r2 in rs:
                        tp2 = r2.m("tpot_p50")
                        if not tp2:
                            continue
                        sp2 = float(r2.g("shared_prefix_tokens", 0))
                        if sp2 <= 0:
                            continue
                        cached = bool(r2.g("prefix_caching", True))
                        pairs.setdefault(sp2, {})[cached] = tp2
                    # Three outcomes, not two. A gap the instrument cannot
                    # resolve is not a disagreement - and the first version
                    # of this check reported one as "PAIRED CHECK FAILED",
                    # which is a different claim entirely.
                    checked, wrong, unresolvable, gaps = 0, [], 0, []
                    for sp2, both in sorted(pairs.items()):
                        if True not in both or False not in both:
                            continue
                        checked += 1
                        measured_gap = both[False] - both[True]
                        predicted_gap = pf * sp2
                        gaps.append((sp2, measured_gap, both[True]))
                        # Below the noise floor of the step itself, the pair
                        # says nothing about any coefficient.
                        # 2% of the step, fixed. self.noise_floor_pct is
                        # not populated until later in fit(), so reading it
                        # here would silently use 0.0 and depend on
                        # initialisation order - which happened to give the
                        # right answer and would stop doing so the day the
                        # order changed.
                        floor = 0.02 * both[True]
                        if abs(measured_gap) < floor:
                            unresolvable += 1
                            continue
                        off = abs(predicted_gap - measured_gap) / abs(measured_gap)
                        if off > 0.30:
                            wrong.append(
                                f"sp{sp2:.0f}: measured {measured_gap:.2f} ms, "
                                f"model {predicted_gap:.2f} ms")
                    if wrong:
                        self.fit_report.append(
                            f"        PAIRED CHECK DISAGREES on "
                            f"{len(wrong)}/{checked} prefix length(s): " +
                            "; ".join(wrong[:3]))
                        self.fit_report.append(
                            "        do not read the two coefficients "
                            "separately; the split is not identified.")
                        self.decode_prompt_suspect = True
                        report_coefficients = [
                            f"        prompt adds {pf:.5f} ms per full-price "
                            f"token + {pc_:.5f} ms per context token "
                            f"(SPLIT UNRELIABLE)"]
                    elif checked and unresolvable == checked:
                        # This is the measured result on this machine, not a
                        # failure: cached and uncached runs at the same
                        # context step at the same rate, to within a few
                        # hundredths of a millisecond on steps of 8-14 ms.
                        # Chunked prefill does not measurably inflate the
                        # decode step here; the context length does.
                        worst = max(abs(g) for _, g, _ in gaps)
                        rel = max(abs(g) / b * 100 for _, g, b in gaps)
                        self.fit_report.append(
                            f"        paired check: cached and uncached runs "
                            f"at the same context differ by at most "
                            f"{worst:.2f} ms ({rel:.1f}% of the step) across "
                            f"{checked} prefix length(s)")
                        self.fit_report.append(
                            f"        -> the full-price term is "
                            f"indistinguishable from zero and is DROPPED. "
                            f"What lengthens the step is the context the "
                            f"decode attends over, whether the cache served "
                            f"it or not")
                        # A coefficient the data cannot see should not be
                        # carried at all. Keeping 0.00002 invites someone to
                        # quote it later as a measurement.
                        self.decode_prompt[key] = (0.0, pc_, base_full, base_ctx)
                        report_coefficients = [
                            f"        prompt adds {pc_:.5f} ms per context "
                            f"token; the full-price term is 0 (not "
                            f"measurable here)"]
                    elif checked:
                        self.fit_report.append(
                            f"        paired check passed on "
                            f"{checked - unresolvable}/{checked} resolvable "
                            f"prefix length(s); {unresolvable} gap(s) were "
                            f"below the noise floor and say nothing")
                        report_coefficients = [
                            f"        prompt adds {pf:.5f} ms per full-price "
                            f"token + {pc_:.5f} ms per context token",
                            f"        -> a cache hit avoids the first and NOT "
                            f"the second: the sequence is as long either way"]
                    if not checked:
                        report_coefficients = [
                            f"        prompt adds {pf:.5f} ms per full-price "
                            f"token + {pc_:.5f} ms per context token "
                            f"(no cached/uncached pair to check the split)"]
                    self.fit_report.extend(report_coefficients)
                    self.fit_report.append(
                        f"        -> TPOT across all {len(errs)} run(s) "
                        f"predicted to {tpot_mape:.1f}% MAPE")
                elif skipped_long:
                    self.fit_report.append(
                        f"        {skipped_long} long-prompt run(s) left out "
                        f"and their effect NOT fitted (needs runs at two or "
                        f"more prompt lengths); TPOT on long prompts will be "
                        f"under-predicted, by 42% in the worst case measured")
            else:
                self.fit_report.append(
                    "        step vs batch NOT fitted (needs 3 distinct batch "
                    "sizes); TPOT will not vary with batch, which finding #37 "
                    "showed can be wrong by 60%")

        # 2. prefill: TTFT against the prompt tokens the cache did NOT remove,
        #    using only runs with a single wave so no queueing is mixed in.
        #
        #    The form is a power law, ttft = c0 + c1 * tokens^alpha, with alpha
        #    found by a coarse search. Measured here: alpha ~ 0.60, so prefill
        #    is SUB-linear in prompt length. A longer prefill does more work
        #    per kernel launch and keeps the GPU busier, so the cost per token
        #    falls as the prompt grows (2.18 ms/token at 38 tokens down to
        #    0.66 ms/token at 2086).
        #
        #    A straight line gives 30% error on this data. A quadratic fits
        #    better (14%) but comes out CONCAVE, and that model predicts TTFT
        #    DECREASING past about 2850 tokens, which is nonsense; it would
        #    silently mislead the planner on long prompts. The power law stays
        #    monotone, so it is safe to extrapolate.
        #    A cached prefix is NOT free: attention still runs over the whole
        #    sequence and the cached KV still has to be read. In this data the
        #    cached runs all carry 38 full-price tokens yet their TTFT spans
        #    70 to 298 ms as the cached prefix grows, so a single curve over
        #    full-price tokens cannot fit both groups. The model therefore
        #    carries a second, much cheaper term for cached tokens.
        #    The fit runs in TWO STAGES, because a single joint fit fails on
        #    this data: most runs carry only the base prompt as full-price
        #    tokens (every cached run, plus the whole concurrency sweep), so
        #    they pin the intercept and leave the slope no leverage. A joint
        #    fit came out at 0.049 ms per token, which prices a 2048-token
        #    uncached prefill at 100 ms and under-predicted it by 45%.
        single = [r for r in usable if r.waves == 1 and r.m("ttft_p50")]

        # Stage 1: the full-price slope, from the runs that actually vary it.
        varying = [r for r in single if r.full_price_tokens > 100]
        spread = len({round(r.full_price_tokens) for r in varying})
        if len(varying) >= 3 and spread >= 3:
            ys = [r.m("ttft_p50") for r in varying]
            best = None
            for i in range(40, 141):                    # alpha from 0.40 to 1.40
                alpha = i / 100
                rows = [[1.0, max(r.full_price_tokens, 1.0) ** alpha]
                        for r in varying]
                try:
                    c0, c1 = least_squares(rows, ys)
                except ValueError:
                    continue
                if c1 <= 0 or c0 < 0:    # must add time; no negative floor
                    continue
                pred = [c0 + c1 * max(r.full_price_tokens, 1.0) ** alpha
                        for r in varying]
                err = mape(ys, pred)
                if best is None or err < best[0]:
                    best = (err, alpha, c0, c1)
            if best:
                err1, self.prefill_alpha, self.prefill_intercept_ms, \
                    self.prefill_coef = best
                self.prefill_form = "power"
                self.fit_report.append(
                    f"prefill (stage 1, {len(varying)} runs with >100 "
                    f"full-price tokens, {spread} lengths)")
                self.fit_report.append(
                    f"  TTFT = {self.prefill_intercept_ms:.0f} ms "
                    f"+ {self.prefill_coef:.3f} * full^{self.prefill_alpha:.2f}"
                    f"   MAPE {err1:.1f}%")
                if self.prefill_alpha < 0.95:
                    self.fit_report.append(
                        "  -> sub-linear: a longer prefill costs less per "
                        "token (better GPU use)")
                elif self.prefill_alpha > 1.05:
                    self.fit_report.append(
                        "  -> super-linear: attention cost grows faster than "
                        "the prompt")

                # Stage 2: what a CACHED token still costs, from the residual
                # on the cached runs. A cached prefix is not free; attention
                # still runs over the whole sequence.
                cached_runs = [r for r in single if r.cached_prefix_tokens > 0]
                if len(cached_runs) >= 2:
                    rows, ys2 = [], []
                    for r in cached_runs:
                        full_part = (self.prefill_intercept_ms
                                     + self.prefill_coef
                                     * max(r.full_price_tokens, 1.0)
                                     ** self.prefill_alpha)
                        rows.append([r.cached_prefix_tokens])
                        ys2.append(r.m("ttft_p50") - full_part)
                    try:
                        (c2,) = least_squares(rows, ys2)
                        self.cached_ms_per_token = max(c2, 0.0)
                    except ValueError:
                        self.cached_ms_per_token = 0.0
                    at1k_full = self.prefill_coef * 1000 ** self.prefill_alpha
                    at1k_cached = self.cached_ms_per_token * 1000
                    self.fit_report.append(
                        f"prefill (stage 2) cached token "
                        f"{self.cached_ms_per_token:.4f} ms "
                        f"({len(cached_runs)} runs)")
                    self.fit_report.append(
                        f"  -> 1000 tokens: {at1k_full:.0f} ms at full price "
                        f"vs {at1k_cached:.0f} ms cached"
                        + (f" ({at1k_full / at1k_cached:.0f}x)"
                           if at1k_cached > 0.5 else " (cached is ~free)"))
            else:
                self.prefill_form = "flat"
        if self.prefill_form not in ("power",):
            # Not enough spread to separate the terms. Fall back to a flat TTFT
            # and say so, rather than inventing a slope the data cannot support.
            self.prefill_intercept_ms = (statistics.median(
                r.m("ttft_p50") for r in single) if single else 0.0)
            self.prefill_coef = 0.0
            self.prefill_alpha = 1.0
            self.cached_ms_per_token = 0.0
            self.prefill_form = "flat"
            self.fit_report.append(
                f"prefill: only {spread} distinct prompt length(s) with TTFT "
                f"data, so no slope can be fitted; using a flat "
                f"{self.prefill_intercept_ms:.0f} ms. Run the prefix_curve "
                f"sweep to fix this.")

        # 3. queue coefficient, LEARNED rather than assumed.
        #
        #    The structure comes from Finding #7: a request that finds no free
        #    batch slot waits for one to open. But how much of a full batch
        #    decode it waits for depends on whether the load is a burst or a
        #    sustained stream, and assuming a whole wave over-predicted
        #    end-to-end latency by 84% on the sustained c16 run.
        #
        #    The magnitude is still fitted rather than assumed:
        #        queue_ms = k * oversubscription * batch_decode_ms
        #    Little's Law says k should be 1. Fitting it is a check, not a
        #    free parameter: a k far from 1 means the data is telling us
        #    something the model does not capture, and that is worth seeing
        #    rather than absorbing.
        over = [r for r in usable
                if r.oversubscription > 0 and r.m("e2e_p50") and r.m("tpot_p50")]
        if over:
            ks = []
            for r in over:
                out = float(r.g("max_tokens", 128))
                tpot = r.m("tpot_p50")
                batch_decode = out * tpot
                service = (self.prefill_intercept_ms
                           + self.prefill_coef
                           * max(r.full_price_tokens, 1.0) ** self.prefill_alpha
                           + self.cached_ms_per_token * r.cached_prefix_tokens
                           + (out - 1) * tpot)
                excess = r.m("e2e_p50") - service
                denom = r.oversubscription * batch_decode
                if denom > 0:
                    ks.append(max(excess / denom, 0.0))
            if ks:
                k = statistics.median(ks)
                # k is the share of a batch decode that a queued request waits
                # for. Above about 2 it stops describing queueing and starts
                # absorbing whatever else went wrong in the measurement, so it
                # is refused rather than quietly used.
                if k > 2.0:
                    self.fit_report.append(
                        f"queue    REFUSED a fitted k of {k:.2f}. A queued "
                        f"request cannot wait for {k:.1f} full batch decodes;")
                    self.fit_report.append(
                        f"         that value is absorbing a measurement "
                        f"problem, not describing queueing. Falling back to "
                        f"k = 1.0 and flagging the latency predictions as "
                        f"unreliable until the data is re-measured.")
                    self.queue_k = 1.0
                    self.queue_suspect = True
                else:
                    self.queue_k = k
                    self.fit_report.append(
                        f"queue    k = {self.queue_k:.2f} "
                        f"(from {len(ks)} oversubscribed run"
                        f"{'s' if len(ks) != 1 else ''}; "
                        f"1.0 would mean a full wave of waiting)")
                    if len(ks) < 3:
                        self.fit_report.append(
                            f"  -> only {len(ks)} run(s) support this; run the "
                            f"'knee' sweep for a firmer estimate")
        else:
            self.fit_report.append(
                "queue    no oversubscribed runs in the ledger; the queue "
                "term is untested. Run the 'knee' sweep.")

        # 4. noise floor: the worst per-config spread seen in THIS data.
        #    Finding #15: this is a property of the session, not the hardware,
        #    so it is taken from the runs at hand and not from a constant.
        noises = [r.noise_pct for r in usable if r.noise_pct]
        self.noise_floor_pct = max(noises) if noises else 0.0
        self.fit_report.append(
            f"noise floor {self.noise_floor_pct:.1f}% "
            f"(worst of {len(noises)} configs; median "
            f"{statistics.median(noises):.1f}%)" if noises else "noise floor unknown")

    # ---- prediction -------------------------------------------------------
    def _correction_for(self, model, quant, dtype) -> tuple:
        """Exact group if known, else the closest fallback, with a note."""
        key = (model, quant, dtype)
        if key in self.decode_correction:
            return self.decode_correction[key], None
        same_model = [(k, v) for k, v in self.decode_correction.items()
                      if k[0] == model]
        if same_model:
            v = statistics.median(v for _, v in same_model)
            return v, (f"no measured run for quantization={quant!r}; using this "
                       f"model's median correction {v:.2f}")
        if self.decode_correction:
            v = statistics.median(self.decode_correction.values())
            return v, (f"no measured run for {model!r}; using the global median "
                       f"correction {v:.2f}. Treat this as a rough estimate.")
        raise ValueError("model has not been fitted")

    def _step_for(self, model: str, quant, dtype: str):
        """
        The batch law for this exact configuration, or the nearest one.

        Falls back across dtype and quantization on purpose. The law
        describes how a decode step grows when more sequences share it -
        KV reads and attention - and that is a property of the batch, not of
        the number format. Without this fallback the planner recommended a
        bfloat16 configuration it had never measured, priced it with the
        flat per-step figure because no bfloat16 batch law existed, and so
        quoted 1123 ms where every measured configuration of the same model
        predicted more. An unmeasured option looked cheapest BECAUSE it was
        unmeasured, which is the worst failure this planner can have.
        """
        key = (model, quant, dtype)
        if key in self.decode_step:
            return self.decode_step[key], None
        same_model = [(k, v) for k, v in self.decode_step.items()
                      if k[0] == model]
        if same_model:
            (k, v) = same_model[0]
            return v, (f"no batch law measured for dtype={dtype}, "
                       f"quantization={quant}; borrowing the one fitted for "
                       f"dtype={k[2]}, quantization={k[1]} of the same model. "
                       f"The step grows with the batch, which is not a "
                       f"property of the number format - but this "
                       f"configuration has not been measured, so verify it "
                       f"before trusting the figure")
        if self.decode_step:
            v = next(iter(self.decode_step.values()))
            return v, ("no batch law measured for this model at all; "
                       "borrowing another model's, which is a guess about "
                       "the shape only. Measure this model before relying "
                       "on the latency figures")
        return None, None

    def predict(self, config: dict) -> dict:
        model = config.get("model")
        quant = config.get("quantization")
        dtype = config.get("dtype", "half")
        # max_tokens is a LIMIT, not a length. These prompts stop at EOS
        # around 116 tokens against a 128 limit, and the prefix runs between
        # 96 and 111, so assuming the limit over-predicted every latency by
        # 9-25% - and, because the queue term is proportional to the output
        # length, inflated TTFT as well. Finding #41.
        #
        # "expected_output_tokens" is what a caller knows or has measured;
        # max_tokens is the fallback and is flagged as an upper bound.
        # TWO lengths, because latency and cost need different statistics.
        #
        # A percentile latency is produced by the request at that percentile,
        # so predicting e2e_p50 needs the MEDIAN completion length. Cost and
        # throughput are total work divided by requests, so they need the
        # MEAN. Finding #41: feeding the mean (116 tokens) into both made
        # e2e worse - 5.5% to 10.5% - while improving cost slightly, which is
        # exactly what mixing up a mean and a percentile looks like.
        #
        # When the median is unknown, the fallback is max_tokens rather than
        # the mean: if most requests run to the limit, the median request is
        # AT the limit, and the mean is dragged below it by the few that stop
        # early. Using the mean there is a known-wrong statistic; using the
        # limit is an upper bound that happens to be close.
        lat_tokens = config.get("expected_output_tokens_p50")
        length_assumed = not lat_tokens
        out_tokens = (float(lat_tokens) if lat_tokens
                      else float(config.get("max_tokens", 128)))
        work_tokens = float(config.get("expected_output_tokens_mean")
                            or out_tokens)
        conc = int(config.get("concurrency", 8))
        mns = int(config.get("max_num_seqs", 8))
        requests = int(config.get("requests", 64))

        r = Run("predict", config, {}, "OK", 0.0)
        correction, note = self._correction_for(model, quant, dtype)
        notes = [note] if note else []
        if length_assumed:
            notes.append(
                f"no measured MEDIAN completion length for this workload, so "
                f"the max_tokens limit of {out_tokens:.0f} is used for the "
                f"latency figures. They are upper bounds: a workload whose "
                f"requests stop early will be faster than this")

        # decode
        theory = roofline_step_ms(model, quant, self.bandwidth_gbps,
                                  config.get("params"))
        tpot = theory * correction

        # The batch that actually shares the step, and what that costs.
        batch = min(conc, mns)
        fitted, step_note = self._step_for(model, quant, dtype)
        if step_note:
            notes.append(step_note)
        # Whether this exact (model, quantization, dtype) was measured, or
        # whether its numbers were borrowed from a neighbour. The planner
        # needs this as a flag rather than by reading the notes: ranking
        # decisions should not depend on matching English.
        borrowed = bool(step_note) or bool(note)
        if fitted:
            c0, c1, alpha = fitted
            batched = c0 + c1 * max(batch, 1) ** alpha
            # The roofline stays a floor: the weights are read once per step
            # however small the batch.
            tpot = max(theory, batched)
            if batch > 1 and batched > theory * correction * 1.1:
                notes.append(
                    f"{batch} sequences share each decode step, which raises "
                    f"it from {theory * correction:.1f} ms to {tpot:.1f} ms")

            # And what the prompt adds on top of the batch. Finding #39.
            prompt_terms = self.decode_prompt.get((model, quant, dtype))
            if not prompt_terms:
                same = [v for k, v in self.decode_prompt.items()
                        if k[0] == model]
                prompt_terms = same[0] if same else None
            if prompt_terms:
                per_full, per_ctx, base_full, base_ctx = prompt_terms
                ctx = r.full_price_tokens + r.cached_prefix_tokens
                extra_full = max(r.full_price_tokens - base_full, 0.0)
                extra_ctx = max(ctx - base_ctx, 0.0)
                add = per_full * extra_full + per_ctx * extra_ctx
                if add > 0:
                    tpot += add
                    notes.append(
                        f"the prompt adds {add:.1f} ms to each decode step: "
                        f"{per_full * extra_full:.1f} ms from "
                        f"{extra_full:.0f} prefill token(s) the cache did "
                        f"not remove, and {per_ctx * extra_ctx:.1f} ms from "
                        f"a {ctx:.0f}-token context the cache cannot shorten")

        # prefill, after the cache has removed what repeats
        ttft_service = (self.prefill_intercept_ms
                        + self.prefill_coef
                        * max(r.full_price_tokens, 1.0) ** self.prefill_alpha
                        + self.cached_ms_per_token * r.cached_prefix_tokens)
        ttft_service = max(ttft_service, 0.0)

        # one batch's decode time, which is what a queued request waits for
        batch_decode_ms = out_tokens * tpot
        queue_ms = self.queue_k * r.oversubscription * batch_decode_ms
        if r.oversubscription > 0:
            notes.append(
                f"client concurrency {conc} exceeds max-num-seqs {mns} by "
                f"{r.oversubscription:.1f}x, so each request waits roughly "
                f"{queue_ms:.0f} ms for a slot (k={self.queue_k:.2f})")
            notes.append(
                "Finding #22: past concurrency == max-num-seqs you gain only "
                "3-7% throughput and p99 doubles. Lower the concurrency "
                "unless you need burst headroom.")

        ttft = ttft_service + queue_ms
        e2e = ttft + (out_tokens - 1) * tpot

        # Throughput: the server decodes at most `mns` sequences at a time,
        # each producing one token per step.
        active = min(conc, mns)
        throughput = active / tpot * 1000.0

        # Cost: wall time for the whole load, amortised per request. This is
        # total work, so it uses the MEAN completion length, not the median.
        wall_s = requests * work_tokens / throughput
        cost_per_1k = (wall_s / 3600) * self.gpu_cost_per_hour / requests * 1000

        band = max(self.noise_floor_pct, 5.0)
        if self.queue_suspect and r.oversubscription > 0:
            notes.append("the queue term could not be fitted from this data; "
                         "treat the latency numbers as a guess until the "
                         "oversubscribed runs are re-measured")
        return {
            "tpot_ms": tpot,
            "ttft_ms": ttft,
            "e2e_ms": e2e,
            "queue_ms": queue_ms,
            "throughput_tok_s": throughput,
            "wall_s": wall_s,
            "cost_per_1k": cost_per_1k,
            "prefill_tokens": r.prefill_tokens,
            "cached_prefix_tokens": r.cached_prefix_tokens,
            "roofline_step_ms": theory,
            "correction": correction,
            "borrowed": borrowed,
            "uncertainty_pct": band,
            "notes": notes,
        }

    # ---- honesty check ----------------------------------------------------
    def validate(self, runs: list) -> dict:
        """
        Leave-one-out: refit without each run, then predict it.

        A model scored on the runs it was fitted to tells you nothing. This
        is the number that belongs in the paper.
        """
        usable = [r for r in runs
                  if r.outcome in ("OK", "PARTIAL") and r.m("tpot_p50")]
        # Validate on the population the model is fitted on, or the score
        # measures the model against runs it was never meant to explain.
        vouched = [r for r in usable if r.controlled]
        if len(vouched) >= 8:
            usable = vouched
        rows = []
        for i, held in enumerate(usable):
            rest = usable[:i] + usable[i + 1:]
            if len(rest) < 2:
                continue
            trial = CostModel(self.gpu, self.bandwidth_gbps, self.gpu_cost_per_hour)
            # Scored TWO ways, because they answer different questions.
            #
            #   "given"   - the model is told how many tokens were actually
            #               generated. This scores the latency model alone.
            #   "assumed" - the model only knows max_tokens, which is what a
            #               planner has before anything runs. This scores the
            #               latency model PLUS the length assumption.
            #
            # Reporting only the first would flatter the model; reporting
            # only the second hides where the error lives. Finding #41 came
            # out of exactly this gap: a 9-25% systematic over-prediction
            # that looked like a latency error and was a length error.
            cfg_given = dict(held.config)
            measured_mean = held.m("out_tokens_mean")
            measured_p50 = held.m("out_tokens_p50")
            if measured_p50:
                cfg_given["expected_output_tokens_p50"] = measured_p50
            if measured_mean:
                cfg_given["expected_output_tokens_mean"] = measured_mean
            measured_len = measured_p50 or measured_mean
            try:
                trial.fit(rest, max_noise_pct=1e9)   # already filtered
                p = trial.predict(cfg_given)
                p_assumed = trial.predict(dict(held.config))
            except ValueError:
                continue
            rows.append({
                "label": held.label,
                "out_tokens_mean": measured_mean,
                "out_tokens_p50": measured_p50,
                "tpot_actual": held.m("tpot_p50"), "tpot_pred": p["tpot_ms"],
                "e2e_actual": held.m("e2e_p50"), "e2e_pred": p["e2e_ms"],
                "cost_actual": held.m("cost_per_1k"), "cost_pred": p["cost_per_1k"],
                "ttft_actual": held.m("ttft_p50"), "ttft_pred": p["ttft_ms"],
                "e2e_pred_assumed": p_assumed["e2e_ms"],
                "ttft_pred_assumed": p_assumed["ttft_ms"],
                "cost_pred_assumed": p_assumed["cost_per_1k"],
            })
        out = {"n": len(rows), "rows": rows}
        for metric in ("tpot", "e2e", "cost", "ttft"):
            a = [r[f"{metric}_actual"] for r in rows if r[f"{metric}_actual"]]
            p = [r[f"{metric}_pred"] for r in rows if r[f"{metric}_actual"]]
            out[f"{metric}_mape"] = mape(a, p)
            # The absolute error as well, because a percentage on a small
            # quantity is mostly a statement about the quantity. Finding #42:
            # TTFT sits at 25-32 ms here, so a 4 ms miss - below any
            # plausible scheduling jitter - reads as 16% and looks like a
            # broken model. The same 4 ms on E2E's ~1000 ms would read as
            # 0.4%. Reporting both stops a percentage from being mistaken
            # for a verdict.
            errs = sorted(abs(x - y) for x, y in zip(a, p))
            out[f"{metric}_mae"] = (
                errs[len(errs) // 2] if errs else float("nan"))
            out[f"{metric}_typical"] = (
                sorted(a)[len(a) // 2] if a else float("nan"))
        for metric in ("e2e", "cost", "ttft"):
            key = f"{metric}_pred_assumed"
            a = [r[f"{metric}_actual"] for r in rows
                 if r[f"{metric}_actual"] and r.get(key)]
            p = [r[key] for r in rows if r[f"{metric}_actual"] and r.get(key)]
            out[f"{metric}_mape_assumed"] = mape(a, p) if a else None
        # Counted separately, because they are used for different things and
        # a column that says "given the real output length" must not be
        # printed when only the mean is on hand: latency then falls back to
        # max_tokens and the column is claiming knowledge it does not have.
        means = [r["out_tokens_mean"] for r in rows if r.get("out_tokens_mean")]
        p50s = [r["out_tokens_p50"] for r in rows if r.get("out_tokens_p50")]
        out["have_lengths"] = len(means) or len(p50s)
        out["have_mean"] = len(means)
        out["have_p50"] = len(p50s)
        out["length_range"] = (min(means), max(means)) if means else None
        return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def gpu_free_total_mib() -> tuple:
    """
    (free, total) VRAM in MiB via nvidia-smi, or (None, None).

    Also lives in runner.py, which needs it before a server starts. Kept here
    so costmodel and planner do not have to import the runner.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True).stdout
        free, total = out.strip().splitlines()[0].split(",")
        return int(free.strip()), int(total.strip())
    except Exception:
        return None, None


def detect_gpu() -> tuple:
    try:
        name = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True
        ).stdout.strip().splitlines()[0].strip()
    except Exception:
        return "unknown", 0.0
    for key, bw in GPU_BANDWIDTH_GBPS.items():
        if key.lower() in name.lower():
            return name, bw
    return name, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="ledger.jsonl")
    ap.add_argument("--gpu", default=None, help="GPU name; auto-detected if omitted")
    ap.add_argument("--bandwidth-gbps", type=float, default=None)
    ap.add_argument("--gpu-cost-per-hour", type=float, default=0.35)
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--explain-decode", action="store_true",
                    help="print the inputs and the predicted step for every "
                         "run, so a wrong coefficient can be seen rather "
                         "than inferred from an aggregate score")
    ap.add_argument("--max-noise-pct", type=float, default=25.0,
                    help="runs whose own passes disagree by more than this "
                         "are excluded from the fit")

    ap.add_argument("--predict", action="store_true",
                    help="price one configuration instead of validating")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--dtype", default="half")
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--shared-prefix-tokens", type=int, default=0)
    ap.add_argument("--no-prefix-caching", action="store_true")
    args = ap.parse_args()

    gpu, bw = (args.gpu, args.bandwidth_gbps or 0.0)
    if not gpu or not bw:
        d_gpu, d_bw = detect_gpu()
        gpu = gpu or d_gpu
        bw = bw or d_bw
    if not bw:
        print(f"unknown memory bandwidth for GPU {gpu!r}.")
        print("Pass --bandwidth-gbps; the roofline term cannot be computed "
              "without it.")
        return 1

    # A missing ledger is the normal state the first time anyone runs this,
    # not a bug, so it gets a sentence instead of a traceback. The first
    # impression a tool makes is the error it shows someone who has done
    # nothing wrong yet.
    try:
        runs = load_runs(Path(args.ledger))
    except FileNotFoundError:
        hint = "inferplan sweep" if __package__ else "python runner.py"
        print(f"  no measurements yet: {args.ledger} does not exist.")
        print(f"  Run a sweep first:  {hint} --sweep knee --reference-every 4")
        return 1
    ok = [r for r in runs if r.outcome in ("OK", "PARTIAL")]
    crashed = [r for r in runs if r.outcome == "CRASH"]

    # Look at the data before learning from it.
    aud = audit(runs, args.max_noise_pct)
    if aud["noisy"] or aud["inversions"] or aud["drift"]:
        print("=" * 74)
        print("  DATA AUDIT  -  these runs do not look like measurements")
        print("=" * 74)
        for r in aud["noisy"]:
            print(f"  noise {r.noise_pct:5.1f}%  {r.label[:52]}")
            print(f"    its own passes disagree by more than "
                  f"{args.max_noise_pct:.0f}%; dropped from the fit")
        for inv in aud["inversions"]:
            print(f"  inversion at concurrency {inv['concurrency']}: "
                  f"{inv['fewer_slots']} slots gave "
                  f"{inv['tput_a']:.0f} tok/s but "
                  f"{inv['more_slots']} slots gave {inv['tput_b']:.0f}")
            print(f"    more batch slots cannot serve less; something outside "
                  f"the config was moving")
        for d in aud["drift"]:
            print(f"  same config measured twice, {d['spread_pct']:.0f}% apart: "
                  f"{d['low']:.0f} vs {d['high']:.0f} tok/s")
            print(f"    {', '.join(l[:34] for l in d['labels'])}")
        print("=" * 74)
        print("  Fit the model only once these are explained. A model will")
        print("  absorb a broken run into a parameter and still look fine.")
        print("=" * 74)
        print()

    cm = CostModel(gpu, bw, args.gpu_cost_per_hour)
    cm.fit(runs, args.max_noise_pct)

    print("=" * 74)
    print(f"  COST MODEL  -  {gpu}  @  {bw:g} GB/s")
    print("=" * 74)
    print(f"  ledger        : {args.ledger}")
    print(f"  runs          : {len(ok)} usable, {len(crashed)} crashed, "
          f"{len(runs)} total")
    print("-" * 74)
    if args.explain_decode:
        print()
        print("=" * 74)
        print("  DECODE STEP, RUN BY RUN")
        print("=" * 74)
        print(f"  {'run':<34}{'batch':>6}{'full':>7}{'ctx':>7}"
              f"{'pred':>8}{'actual':>8}{'err':>7}")
        print("-" * 74)
        for r in sorted(runs, key=lambda x: x.label):
            tp = r.m("tpot_p50")
            if not tp or r.outcome not in ("OK", "PARTIAL"):
                continue
            try:
                pred = cm.predict(r.config)["tpot_ms"]
            except ValueError:
                continue
            batch = min(int(r.g("concurrency", 8)),
                        int(r.g("max_num_seqs", 8)))
            ctx = r.full_price_tokens + r.cached_prefix_tokens
            print(f"  {r.label[:33]:<34}{batch:>6}"
                  f"{r.full_price_tokens:>7.0f}{ctx:>7.0f}"
                  f"{pred:>8.2f}{tp:>8.2f}{(pred - tp) / tp * 100:>+6.1f}%")
        print("=" * 74)
        print("  'full' is the prompt tokens the cache did NOT remove;")
        print("  'ctx' is the whole sequence the decode attends over.")
        print("  Two runs with the same ctx and different full isolate the")
        print("  chunked-prefill term; if the model gives them the same")
        print("  prediction, that term is wrong however good the MAPE looks.")
        print("=" * 74)

    for line in cm.fit_report:
        print(f"  {line}")
    print("=" * 74)

    if crashed:
        print()
        print("  infeasible configurations learned from crashes:")
        for r in crashed:
            print(f"    {r.label[:52]:<54} {r.config.get('crash_kind', '')}")

    if args.predict:
        cfg = {"model": args.model, "quantization": args.quantization,
               "dtype": args.dtype, "max_num_seqs": args.max_num_seqs,
               "concurrency": args.concurrency, "requests": args.requests,
               "max_tokens": args.max_tokens,
               "shared_prefix_tokens": args.shared_prefix_tokens,
               "prefix_caching": not args.no_prefix_caching}
        p = cm.predict(cfg)
        print()
        print("=" * 74)
        print("  PREDICTION")
        print("=" * 74)
        print(f"  roofline step   : {p['roofline_step_ms']:8.2f} ms "
              f"(x {p['correction']:.2f} correction)")
        print(f"  TPOT            : {p['tpot_ms']:8.2f} ms")
        print(f"  prefill tokens  : {p['prefill_tokens']:8.0f} "
              f"({p['cached_prefix_tokens']:.0f} removed by the cache)")
        print(f"  TTFT            : {p['ttft_ms']:8.0f} ms"
              + (f"  (incl. {p['queue_ms']:.0f} ms queueing)"
                 if p["queue_ms"] else ""))
        print(f"  E2E             : {p['e2e_ms']:8.0f} ms")
        print(f"  throughput      : {p['throughput_tok_s']:8.0f} tokens/s")
        print(f"  cost per 1K     : ${p['cost_per_1k']:.4f}"
              f"   +/- {p['uncertainty_pct']:.0f}%")
        for n in p["notes"]:
            print(f"  note            : {n}")
        print("=" * 74)
        return 0

    if args.no_validate:
        return 0

    v = cm.validate(runs)
    print()
    print("=" * 74)
    print(f"  LEAVE-ONE-OUT VALIDATION  ({v['n']} runs held out one at a time)")
    print("=" * 74)
    have = v.get("have_lengths") or 0
    have_p50 = v.get("have_p50") or 0
    have_mean = v.get("have_mean") or 0
    if have:
        left = ("measured length" if have_p50 else "measured MEAN")
        print(f"  {'':<14}{left:>16}{'assuming':>12}")
        print(f"  {'':<14}{'(median+mean)' if have_p50 else '(latency: none)':>16}"
              f"{'max_tokens':>12}")
        print("-" * 74)
    for metric, label in (("tpot", "TPOT"), ("ttft", "TTFT"),
                          ("e2e", "E2E latency"), ("cost", "cost per 1K")):
        m = v.get(f"{metric}_mape", float("nan"))
        verdict = "" if m != m else ("  ok" if m <= 15 else "  OVER TARGET")
        mae = v.get(f"{metric}_mae", float("nan"))
        typ = v.get(f"{metric}_typical", float("nan"))
        # Cost is a fraction of a cent, so two decimals round it to zero and
        # the line reads "median miss 0.00 on a typical 0.0" - a number that
        # says nothing printed next to a verdict. Each metric gets the
        # precision its magnitude needs.
        if metric == "cost":
            side = ("" if mae != mae
                    else f"   median miss ${mae:.5f} on a typical ${typ:.4f}")
        else:
            side = ("" if mae != mae
                    else f"   median miss {mae:.2f} ms on a "
                         f"typical {typ:.1f} ms")
        if have and v.get(f"{metric}_mape_assumed") is not None:
            ma = v[f"{metric}_mape_assumed"]
            print(f"  {label:<14}{m:>13.1f} %{ma:>11.1f} %{verdict}")
        else:
            print(f"  {label:<14}{m:>13.1f} %{'':>13}{verdict}")
        print(f"  {'':<14}{side}")
    print("-" * 74)
    print(f"  target is 15% or better; the session noise floor is "
          f"{cm.noise_floor_pct:.1f}%,")
    print(f"  so anything under that is measurement, not model error.")
    if have:
        lo, hi = v["length_range"] or (0, 0)
        limit = max(int(r.g("max_tokens", 128)) for r in runs)
        print()
        print(f"  Completions here ran {lo:.0f}-{hi:.0f} tokens on average "
              f"against a {limit}-token limit.")
        if have_p50:
            print(f"  LEFT column: latency predicted from the measured MEDIAN "
                  f"length and")
            print(f"  cost from the measured mean - each statistic where it "
                  f"belongs (#41).")
            print(f"  RIGHT column: what a planner has before anything runs.")
        else:
            print(f"  Only the MEAN is recorded in this ledger "
                  f"({have_mean} run(s)); no median.")
            print(f"  A percentile latency needs a percentile length, so the "
                  f"latency rows")
            print(f"  fall back to max_tokens and the two columns are "
                  f"IDENTICAL for them.")
            print(f"  Only cost differs, because cost is total work and the "
                  f"mean is right")
            print(f"  for that. Re-run a sweep to record the median (#41).")
    elif cm.n_runs:
        print()
        print(f"  No run in this ledger recorded its completion length, so")
        print(f"  every prediction assumes max_tokens and is an upper bound.")
        print(f"  Re-run a sweep with a current bench.py to record it.")
    print("=" * 74)

    worst = sorted(v["rows"],
                   key=lambda r: -abs((r["e2e_pred"] - (r["e2e_actual"] or 1))
                                      / (r["e2e_actual"] or 1)))[:5]
    if worst:
        print()
        print("  worst five on end-to-end latency:")
        # Labels are shortened from the FRONT, not the back. Truncating the
        # tail made fp16_half_b8_c8_len4096_u0.7_kvauto_sp512_nopc and
        # ..._kvauto display as the same run, so a reader comparing two rows
        # was comparing names that were not the ones measured. The part that
        # distinguishes these configurations is always the suffix.
        labels = [r["label"] for r in worst if r.get("e2e_actual")]
        width = 46
        def short(label):
            return label if len(label) <= width else "..." + label[-(width - 3):]
        print(f"    {'config':<{width + 1}}{'actual':>9}{'pred':>9}{'err':>8}")
        for r in worst:
            a, p = r["e2e_actual"], r["e2e_pred"]
            if not a:
                continue
            print(f"    {short(r['label']):<{width + 1}}{a:>9.0f}{p:>9.0f}"
                  f"{(p / a - 1) * 100:>+7.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())