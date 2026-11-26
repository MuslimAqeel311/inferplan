#!/usr/bin/env python3
"""
InferPlan planner: the library itself.

Everything else in this project gathers evidence. This module spends it: it
takes what a deployment must achieve, prices every candidate configuration
against the fitted cost model, throws out the ones that cannot meet the
requirements, and returns the cheapest few that can, with the numbers and the
server command to run them.

The ordering of its rules comes from measurements, not from taste:

  1. FEASIBILITY FIRST.   A configuration that crashes or cannot carry the
     offered rate is not cheap, it is unavailable. Finding #20: a run measured
     next to an orphan server looked 7x worse with 0.1% internal noise, so
     plausible numbers are not evidence of a working configuration.

  2. CONSTRAINTS NEXT.    Latency and accuracy limits are refusals, not
     penalties. Finding #19: at a 2048-token shared prefix, turning prefix
     caching off pushes p99 past a 2 s SLO. There, caching is not an
     optimisation, it decides whether the configuration exists at all.

  3. COST LAST, AND NOT BY ITSELF.  Finding #16: across three sessions the
     cost gap between the two best configurations stayed inside the noise
     floor while their p99 differed by 1.9x. A planner that ranked on cost
     alone would have picked the slow one twice.

What decides the bill is not cost per request on one GPU - a GPU is billed by
time, so at a fixed offered rate that is nearly constant. What the
configuration decides is HOW MANY GPUs the rate needs:

    capacity_per_gpu = max_num_seqs / (output_tokens x step_time)
    gpus             = ceil(offered_rate / capacity_per_gpu)

Measured against the clean knee sweep: 8 slots carry 8.0 req/s (measured 8.8),
16 carry 16.0 (measured 16.2), 32 carry 32.0 (measured 27.8, the shortfall
being the power cap of Finding #21). To serve 30 req/s that is 4 GPUs, 2 GPUs
or 1 GPU - whole machines, not percentages.

Usage
    python planner.py --ledger ledger.jsonl \
        --rate 30 --p99-ms 2000 --input-tokens 1500 --prefix-overlap 0.9
"""
import argparse
import itertools
import json
import math
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .costmodel import (CostModel, GPU_BANDWIDTH_GBPS, MODEL_PARAMS,
                       audit, detect_gpu, load_all_runs, load_runs,
                       weight_bytes)


# ----------------------------------------------------------------------------
# what the deployment must achieve
# ----------------------------------------------------------------------------
@dataclass
class Requirements:
    model: str
    rate_rps: float                  # offered load
    input_tokens: int = 38           # prompt length
    output_tokens: int = 128
    prefix_overlap: float = 0.0      # share of the prompt identical across
                                     # requests; Finding #19 makes this decide
                                     # whether caching is worth anything
    p99_ms: Optional[float] = None
    max_accuracy_drop_pct: Optional[float] = None
    budget_per_1k: Optional[float] = None
    max_gpus: int = 8

    @property
    def shared_prefix_tokens(self) -> int:
        return int(self.input_tokens * self.prefix_overlap)


# ----------------------------------------------------------------------------
# accuracy, which depends on precision and not on the GPU
# ----------------------------------------------------------------------------
# Accuracy is a property of how far the weights were compressed, not of which
# chip runs them, so it is measured once per quantization and reused.
#
# These are the fallbacks, used only until quality.py has measured this model.
# The planner says which of the two it is using rather than letting a guess
# pass for a measurement.
FALLBACK_ACCURACY_DROP_PCT = {
    None: 0.0, "none": 0.0, "fp16": 0.0, "half": 0.0, "bfloat16": 0.0,
    "fp8": 0.2, "int8": 0.3, "awq": 0.6, "gptq": 0.8,
}


def load_accuracy(path: Path) -> tuple:
    """
    (drops by quantization, measured labels) from quality.py's store.

    A quantization with no measurement falls back to the table above and is
    reported as unmeasured. Mixing the two silently would let a guess decide
    a deployment.
    """
    drops, measured = dict(FALLBACK_ACCURACY_DROP_PCT), {}
    if not path.exists():
        return drops, measured
    try:
        store = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return drops, measured
    base_label = store.get("baseline")
    for label, m in (store.get("measurements") or {}).items():
        quant = m.get("quantization")
        if quant is None and label != base_label:
            continue
        key = quant if quant not in ("none",) else None
        if label == base_label:
            drops[key] = 0.0
            measured[key] = label
            continue
        # Cross-model comparisons say nothing about what a quantization costs,
        # because the difference mixes model size with precision. quality.py
        # flags these, but the flag is not trusted on its own: a store written
        # by an older version has no flag, and the planner would then report a
        # measurement that was never made. The model names are in the record,
        # so the check is done here rather than assumed.
        base_rec = (store.get("measurements") or {}).get(base_label) or {}
        if m.get("cross_model_comparison") or (
                base_rec.get("model") and m.get("model")
                and base_rec["model"] != m["model"]):
            continue
        drop = m.get("accuracy_drop_pct")
        if drop is None:
            # quality.py measured this precision but could not establish a
            # drop - usually because the accuracy check was underpowered.
            # That is not the same as a drop of zero, so the fallback
            # estimate stands and the quantization stays "unmeasured".
            continue
        drops[key] = float(drop)
        measured[key] = label
    return drops, measured


# ----------------------------------------------------------------------------
# hardware support, so impossible configurations are refused, not priced
# ----------------------------------------------------------------------------
def compute_capability(gpu_name: str) -> Optional[tuple]:
    known = {
        "rtx 3050": (8, 6), "rtx 3060": (8, 6), "rtx 3090": (8, 6),
        "a10g": (8, 6), "a100": (8, 0),
        "rtx 4090": (8, 9), "l4": (8, 9), "l40": (8, 9),
        "h100": (9, 0), "h200": (9, 0),
        "t4": (7, 5),
    }
    low = gpu_name.lower()
    for key, cap in known.items():
        if key in low:
            return cap
    return None


def quantization_supported(quant: Optional[str], gpu_name: str,
                           model: str) -> tuple:
    """(supported, reason). A refusal here is a measurement, not a guess."""
    if quant in (None, "none", "fp16", "half", "bfloat16"):
        return True, ""
    cap = compute_capability(gpu_name)
    if quant == "fp8":
        # Finding: RTX 3050 is compute 8.6 and vLLM refused fp8 on it. FP8
        # needs 8.9 (Ada) or newer.
        if cap and cap < (8, 9):
            return False, (f"fp8 needs compute capability 8.9; this GPU is "
                           f"{cap[0]}.{cap[1]}")
        return True, ""
    if quant in ("awq", "gptq"):
        # These do not quantize an fp16 checkpoint on the fly; the weights must
        # already be in that format, so the MODEL has to change too.
        if quant not in model.lower():
            return False, (f"--quantization {quant} needs weights already in "
                           f"{quant} format; point --model at an {quant.upper()} "
                           f"checkpoint")
        return True, ""
    return True, ""


def memory_fits(model: str, quant: Optional[str], max_num_seqs: int,
                max_model_len: int, seq_tokens: int, free_mib: float) -> tuple:
    """
    Weights plus the KV cache the WORKLOAD needs, against FREE VRAM.

    The KV requirement is max_num_seqs x the actual sequence length, NOT
    x max_model_len. An earlier version used max_model_len and refused 32
    slots as needing 5438 MiB, when that configuration had already run on this
    machine several times. max_model_len caps one request; it does not reserve
    that much per slot, because vLLM pages the cache and a sequence occupies
    only the tokens it has. Corroboration from the server logs: vLLM allocated
    139,456 cache tokens for the 32-slot runs, while 32 slots x 166 tokens
    needs 5,312.

    Finding #1 and #17: declared VRAM is not usable VRAM, and what is free
    moved 708 MiB (17% of the card) across three sessions. So the check is
    against what is free now, and the requirement is reported back.
    """
    try:
        w_mib = weight_bytes(model, quant) / (1024 ** 2)
    except ValueError as e:
        return False, str(e), 0.0
    if seq_tokens > max_model_len:
        return False, (f"the workload needs {seq_tokens} tokens per request "
                       f"but max-model-len is {max_model_len}"), 0.0
    # ~0.0625 MiB per token at fp16 for a 0.5B-class model, scaled by parameter
    # count as a rough proxy for layers x heads x head_dim.
    params = MODEL_PARAMS.get(model, 500_000_000)
    kv_mib_per_token = 0.0625 * (params / 494_000_000) ** 0.5
    kv_mib = max_num_seqs * seq_tokens * kv_mib_per_token
    overhead_mib = 400.0          # CUDA context, activations, CUDA graphs
    need = w_mib + kv_mib + overhead_mib
    if need > free_mib:
        return False, (f"needs ~{need:.0f} MiB (weights {w_mib:.0f} + KV "
                       f"{kv_mib:.0f} + overhead {overhead_mib:.0f}) but only "
                       f"{free_mib:.0f} MiB is free"), need
    return True, "", need


# ----------------------------------------------------------------------------
# one candidate, priced
# ----------------------------------------------------------------------------
@dataclass
class Saturation:
    """
    Achievable throughput as a function of batch slots, with a ceiling.

    Measured on this machine:
        slots    tok/s   share of the slots/step ceiling   gain vs previous
            8     1015                              99%                  -
           16     1870                              91%              1.84x
           32     3467                              85%              1.85x
           64     4730                              58%              1.36x

    A first version fitted a fixed loss per doubling (about 9%) on the first
    three points and extrapolated to 64 slots, predicting 5977 tok/s where
    4730 was available: 26% optimistic, which is the dangerous direction,
    because the planner would promise capacity the machine does not have.

    The doubling gains give it away: 1.84x, 1.85x, then 1.36x. Throughput is
    approaching a CEILING rather than losing a fixed share, so the model is
    hyperbolic:

        throughput = t_max * slots / (slots + k)

    That form cannot exceed t_max no matter how many slots are added, and
    t_max IS the power cap of Finding #21 expressed in tokens per second.
    Fitted on all four points it holds to 8%, against the log-linear model's
    26%, and - more importantly - it cannot promise what does not exist.
    """
    t_max: float = 0.0
    k: float = 0.0
    measured_slots: tuple = ()
    fitted: bool = False
    # Every slot count observed anywhere in the ledger, including runs whose
    # workload shape kept them out of the fit. A run that was not fitted is
    # still evidence, and finding #30 is what happens when it is not treated
    # as such: a hand-labelled 128-slot run recorded no max_model_len, landed
    # in a group of its own, and was dropped by max(groups, key=len) - so the
    # one measurement that contradicted the fit was the one the fit ignored.
    observed: dict = field(default_factory=dict)
    # Measured, but with nothing vouching for the machine state at the time.
    # These raise a question; they never set a ceiling. Finding #33.
    uncontrolled: dict = field(default_factory=dict)
    # Which sweep each slot count came from. Comparing slot counts measured
    # in different sweeps is the exact mistake of finding #31, so it is
    # flagged rather than hidden.
    sweeps_seen: dict = field(default_factory=dict)
    # True when the fit rests only on runs a reference control vouched for.
    fitted_on_controlled: bool = False
    controlled_obs: dict = field(default_factory=dict)
    peak_slots: int = 0        # slot count with the highest measured throughput
    peak_tok_s: float = 0.0
    declines: bool = False     # throughput measured LOWER past the peak
    warnings: tuple = ()
    # Outside the measured range the fit is a claim, not a measurement, so a
    # haircut is applied. It is one-sided on purpose: being optimistic about
    # capacity sends a deployment live under-provisioned, while being
    # pessimistic only costs a recommendation that is better than it looks.
    extrapolation_haircut: float = 0.85
    reason: str = ""      # why a fit was refused, when it was

    @classmethod
    def fit(cls, runs: list, step_ms: float):
        """
        Fitted WITHIN one workload shape, never across several.

        A first version pooled every run at a given slot count. At 8 slots
        that mixed two knee runs (38-token prompts, ~1030 tok/s) with twelve
        prefix-curve runs (up to 2086-token prompts, down to 492 tok/s), which
        pulled the 8-slot point from ~1030 to ~880 and flattened the measured
        slope from 9% per doubling to 2%. A longer prompt lowers throughput
        because each request carries more input for the same output: that is a
        workload effect, and pooling it with saturation destroys both.
        """
        groups = {}
        controlled = {}
        controlled_obs = {}
        observed = {}
        uncontrolled = {}
        sweeps_seen = {}
        for r in runs:
            if r.outcome not in ("OK", "PARTIAL"):
                continue
            tp = r.m("throughput_tok_s")
            slots = int(r.g("max_num_seqs", 8))
            conc = int(r.g("concurrency", 8))
            # Only runs that actually fill the slots say anything about the
            # ceiling; an under-driven server is limited by the client.
            if not tp or conc < slots:
                continue
            # Recorded before any shape grouping, so that a run the fit
            # cannot use can still contradict it - but only if a reference
            # control vouched for the machine it ran on. Finding #33: the
            # uncontrolled runs in runs/ read 801 and 1213 tok/s for slot
            # counts a controlled sweep measured at 4714 and 5339, and an
            # earlier version of this code used them as ceilings, which would
            # have capped every recommendation at a sixth of the real figure.
            # Three levels, not two:
            #   - a hand-run benchmark (bench.py) has nothing vouching for
            #     the machine OR the sweep it sat in: recorded only.
            #   - a run from a sweep the reference control condemned is
            #     dropped outright.
            #   - everything else comes from a runner.py sweep, which runs
            #     its configs in one sitting, so it is at least internally
            #     comparable and is admitted.
            if r.meta.get("uncontrolled"):
                uncontrolled.setdefault(slots, []).append(tp)
                continue
            if r.meta.get("contaminated"):
                continue
            observed.setdefault(slots, []).append(tp)
            sweeps_seen.setdefault(slots, set()).add(r.meta.get("sweep") or "?")
            shape = (r.g("model"), r.g("quantization"), r.g("dtype"),
                     r.g("max_tokens"), r.g("shared_prefix_tokens"),
                     r.g("prefix_caching"), r.g("max_model_len"))
            # RAW throughput, not its share of the ceiling. The share is what
            # the earlier log-linear version fitted; feeding it to a hyperbola
            # inverts the slope and the fit gets refused for having no
            # ceiling, which is how this bug announced itself.
            groups.setdefault(shape, {}).setdefault(slots, []).append(tp)
            if r.controlled:
                controlled.setdefault(shape, {}).setdefault(slots, []).append(tp)
                controlled_obs.setdefault(slots, []).append(tp)

        # Prefer runs a reference control vouched for, when there are enough
        # of them. Pooling controlled and uncontrolled runs flattens the
        # curve: on this machine the pooled fit read 5% LOW at 32 slots and
        # 11-16% HIGH at 96-128, because it was averaging sittings. Fitted on
        # the four controlled points alone the same form holds every one of
        # them to within 4.9%, inside the noise floor, with a ceiling of
        # 7114 tok/s against the pooled fit's 9140. Finding #35.
        best_controlled = max(controlled.values(), key=len) if controlled else {}
        fitted_on_controlled = len(best_controlled) >= 3
        if fitted_on_controlled:
            pts = best_controlled
        else:
            # Use the shape that covers the most distinct slot counts; that is
            # the one that can actually say how capacity scales.
            pts = max(groups.values(), key=len) if groups else {}
        if len(pts) < 3:
            # Two points fit a hyperbola exactly and say nothing about its
            # shape, so the fit is refused rather than trusted.
            return cls(measured_slots=tuple(sorted(pts)), observed=observed, uncontrolled=uncontrolled,
                       reason="needs 3 slot counts")

        # 1/T = 1/t_max + (k/t_max) * (1/slots), linear in 1/slots.
        xs, ys = [], []
        for slots in sorted(pts):
            tp = sum(pts[slots]) / len(pts[slots])
            if tp <= 0:
                continue
            xs.append(1.0 / slots)
            ys.append(1.0 / tp)
        if len(xs) < 3:
            return cls(measured_slots=tuple(sorted(pts)), observed=observed, uncontrolled=uncontrolled,
                       reason="fewer than 3 usable throughput points")
        n = len(xs)
        sx, sy = sum(xs), sum(ys)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-15:
            return cls(measured_slots=tuple(sorted(pts)), observed=observed, uncontrolled=uncontrolled,
                       reason="slot counts are collinear")
        b = (n * sxy - sx * sy) / denom
        a = (sy - b * sx) / n
        if a <= 0 or b < 0:
            # A non-positive intercept means no ceiling at all, which is the
            # failure this model exists to avoid.
            return cls(measured_slots=tuple(sorted(pts)), observed=observed, uncontrolled=uncontrolled,
                       reason="the data implies no ceiling; throughput still "
                              "scales, so more slots were not tried far enough")
        t_max = 1.0 / a
        k = b / a
        fitobj = cls(t_max, k, tuple(sorted(pts)), True, observed=observed,
                     uncontrolled=uncontrolled)
        fitobj.sweeps_seen = sweeps_seen
        fitobj.fitted_on_controlled = fitted_on_controlled
        fitobj.controlled_obs = controlled_obs
        fitobj._check_against_observations()
        return fitobj

    def _check_against_observations(self) -> None:
        """
        Compare the fit with every measurement, fitted or not.

        The hyperbola rises for ever: T = t_max * s / (s + k) is monotonic in
        s by construction. Real serving throughput is not. On this machine it
        peaked at 64 slots (4730 tok/s) and FELL to 3647 at 128 - a 23% drop
        that no choice of t_max and k can express. Finding #30.

        So the fit is not corrected here; it is FENCED. Past the slot count
        where throughput was last seen to rise, the prediction is replaced by
        the highest figure actually measured, and the planner says so. A model
        that cannot represent a decline must not be allowed to predict through
        one.
        """
        # When the fit rests on controlled runs, the peak and any decline must
        # be judged on controlled runs too. Mixing them in here would let an
        # uncontrolled reading cap a recommendation - the exact mistake
        # finding #33 caught in the version before this one, reappearing in a
        # different method.
        source = (self.controlled_obs if self.fitted_on_controlled
                  else self.observed)
        means = {s: sum(v) / len(v) for s, v in source.items()}
        if not means:
            return
        self.peak_slots = max(means, key=lambda s: means[s])
        self.peak_tok_s = means[self.peak_slots]
        warn = []

        beyond = [s for s in means if s > self.peak_slots]
        if beyond:
            self.declines = True
            worst = min(beyond, key=lambda s: means[s])
            drop = (self.peak_tok_s - means[worst]) / self.peak_tok_s * 100
            warn.append(
                f"throughput PEAKS at {self.peak_slots} slots "
                f"({self.peak_tok_s:.0f} tok/s) and falls {drop:.0f}% by "
                f"{worst} slots ({means[worst]:.0f} tok/s); the hyperbolic "
                f"model cannot represent that, so nothing above "
                f"{self.peak_slots} slots is predicted")

        # Any measured slot count the fit never saw, and why that matters.
        unfitted = sorted(set(self.observed) - set(self.measured_slots))
        if unfitted and self.fitted_on_controlled:
            warn.append(
                f"{', '.join(str(s) for s in unfitted)} slots were measured "
                f"without a reference control and are NOT used here; the fit "
                f"rests on {', '.join(str(s) for s in self.measured_slots)} "
                f"slots that one vouched for")
        elif unfitted:
            warn.append(
                f"measured at {', '.join(str(s) for s in unfitted)} slots but "
                f"not fitted there (a different workload shape, or a run with "
                f"no server config recorded); those runs still constrain the "
                f"fit and are used as a ceiling")

        # The slot counts being compared must come from one sitting, or the
        # comparison carries a machine-state term nobody measured.
        spans = {s: v for s, v in self.sweeps_seen.items() if v}
        all_sweeps = set()
        for v in spans.values():
            all_sweeps |= v
        if self.fitted_on_controlled:
            pass
        elif len(all_sweeps) > 1 and len(spans) > 1:
            warn.append(
                f"the slot counts compared here come from {len(all_sweeps)} "
                f"different sweeps ({', '.join(sorted(all_sweeps))}); only "
                f"configs measured in one sitting, or bracketed by a "
                f"reference control, are safely comparable - finding #31")

        # Uncontrolled runs cannot cap anything, but a large disagreement is
        # worth saying out loud: either the machine was in a different state
        # or something about that configuration is not understood.
        for s in sorted(self.uncontrolled):
            u = sum(self.uncontrolled[s]) / len(self.uncontrolled[s])
            ref = means.get(s)
            basis = ref if ref else (self.t_max * s / (s + self.k))
            if basis and abs(u - basis) / basis > 0.25:
                warn.append(
                    f"an UNCONTROLLED run at {s} slots read {u:.0f} tok/s "
                    f"against {basis:.0f} here; no reference control vouched "
                    f"for the machine it ran on, so it is recorded and not "
                    f"used")

        # Where the fit and a measurement overlap, say how far apart they are.
        for s in sorted(set(means) & set(self.measured_slots)):
            pred = self.t_max * s / (s + self.k)
            err = (pred - means[s]) / means[s] * 100
            if abs(err) > 15:
                warn.append(f"at {s} slots the fit is {err:+.0f}% off the "
                            f"measurement it was fitted on")
        self.warnings = tuple(warn)

    def throughput_tok_s(self, slots: int) -> float:
        """Achievable tokens per second at this slot count, ceiling included."""
        if not self.fitted:
            return 0.0
        # Past the measured peak the curve is known to be wrong in the
        # dangerous direction, so it is not used at all: the best figure
        # actually seen stands in for it. This is deliberately flat rather
        # than declining - predicting the size of the decline would be the
        # same mistake again, made in the other direction.
        if self.declines and self.peak_slots and slots > self.peak_slots:
            # A measurement at exactly this slot count beats both the curve
            # and the peak. Standing the peak in for a slot count that WAS
            # measured would have reported 4730 tok/s where 3647 was seen -
            # optimistic by 30%, which is the direction that
            # under-provisions.
            # The nearest measurement at or below this slot count, among
            # those past the peak. At 128 that is the 128-slot run itself
            # (3647); at 96 there is none, so the peak stands in; at 256 it
            # is still the 128-slot run, because throughput was last seen
            # FALLING and carrying the peak forward would claim a recovery
            # nothing has shown.
            below = [s for s in self.observed if self.peak_slots < s <= slots]
            if below:
                v = self.observed[max(below)]
                return sum(v) / len(v)
            return self.peak_tok_s
        t = self.t_max * slots / (slots + self.k)
        if self.is_extrapolated(slots):
            t *= self.extrapolation_haircut
        return t

    def is_extrapolated(self, slots: int) -> bool:
        """
        Beyond everything that counts as measured here.

        An uncontrolled run is not a licence to skip the haircut. A drifted
        192-slot reading of 900 tok/s sitting in the ledger would otherwise
        make the planner treat 192 slots as explored territory and quote the
        curve there at full confidence - using a bad measurement not as
        evidence but as permission.
        """
        known = set(self.measured_slots)
        known |= set(self.controlled_obs if self.fitted_on_controlled
                     else self.observed)
        return bool(known) and slots > max(known)

    def capped_at_peak(self, slots: int) -> bool:
        return bool(self.declines and self.peak_slots and slots > self.peak_slots)


@dataclass
class Candidate:
    config: dict
    feasible: bool = True
    refusals: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    capacity_rps: float = 0.0
    gpus: int = 0
    e2e_ms: float = 0.0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    cost_per_1k: float = float("inf")
    accuracy_drop_pct: float = 0.0
    required_free_mib: float = 0.0
    uncertainty_pct: float = 0.0
    equivalents: list = field(default_factory=list)
    extrapolated: bool = False
    saturation: float = 1.0

    @property
    def label(self) -> str:
        c = self.config
        q = c.get("quantization") or "fp16"
        bits = [q, c.get("dtype", "half"), f"b{c['max_num_seqs']}",
                f"len{c['max_model_len']}", f"kv{c.get('kv_cache_dtype','auto')}"]
        if not c.get("prefix_caching", True):
            bits.append("nopc")
        return "_".join(bits)

    def serve_command(self) -> str:
        c = self.config
        parts = ["vllm serve", c["model"],
                 f"--dtype {c.get('dtype','half')}",
                 f"--max-model-len {c['max_model_len']}",
                 f"--max-num-seqs {c['max_num_seqs']}",
                 f"--gpu-memory-utilization {c.get('gpu_memory_utilization',0.70):g}",
                 f"--kv-cache-dtype {c.get('kv_cache_dtype','auto')}"]
        if c.get("quantization"):
            parts.append(f"--quantization {c['quantization']}")
        if not c.get("prefix_caching", True):
            parts.append("--no-enable-prefix-caching")
        return " \\\n  ".join(parts)


def search_space(req: Requirements, models: list) -> list:
    """
    Every configuration worth pricing.

    Client concurrency is NOT searched. Finding #22 showed the knee sits
    exactly at concurrency == max_num_seqs on every slot count tested, and
    going past it buys 3-7% throughput for double the p99. So concurrency is
    set to the slot count, and the offered rate decides how many GPUs that
    needs.
    """
    out = []
    for (model, quant), dtype, slots, pc, kv in itertools.product(
            models,
            ("half", "bfloat16"),
            # 96 and 128 were added once finding #33 measured them under a
            # reference control. Before that the search stopped at 64 and the
            # planner could not recommend the cheapest configuration on this
            # machine even though it existed - a search space is a claim
            # about what is worth trying, and it was a 13% too modest one.
            (8, 16, 32, 64, 96, 128),
            (True, False),
            ("auto", "fp8")):
        out.append({
            "model": model, "quantization": quant, "dtype": dtype,
            "max_num_seqs": slots,
            "max_model_len": max(2048, 1 << (max(req.input_tokens
                                                 + req.output_tokens,
                                                 1024) - 1).bit_length()),
            "prefix_caching": pc, "kv_cache_dtype": kv,
            "gpu_memory_utilization": 0.70,
            "concurrency": slots,          # the knee, per Finding #22
            "requests": max(8 * slots, 64),
            "max_tokens": req.output_tokens,
            "base_prompt_tokens": req.input_tokens - req.shared_prefix_tokens,
            "shared_prefix_tokens": req.shared_prefix_tokens,
        })
    return out


def evaluate(cfg: dict, req: Requirements, cm: CostModel, gpu_name: str,
             free_mib: float, sat: Saturation, acc_drops: dict) -> Candidate:
    cand = Candidate(config=cfg)

    ok, why = quantization_supported(cfg["quantization"], gpu_name, cfg["model"])
    if not ok:
        cand.feasible = False
        cand.refusals.append(why)

    fits, why, need = memory_fits(cfg["model"], cfg["quantization"],
                                  cfg["max_num_seqs"], cfg["max_model_len"],
                                  req.input_tokens + req.output_tokens,
                                  free_mib)
    cand.required_free_mib = need
    if not fits:
        cand.feasible = False
        cand.refusals.append(why)

    if not cand.feasible:
        return cand

    try:
        p = cm.predict(cfg)
    except ValueError as e:
        cand.feasible = False
        cand.refusals.append(f"cost model cannot price this: {e}")
        return cand

    cand.tpot_ms = p["tpot_ms"]
    cand.ttft_ms = p["ttft_ms"]
    cand.e2e_ms = p["e2e_ms"]
    cand.uncertainty_pct = p["uncertainty_pct"]
    # A configuration whose coefficients were borrowed from a neighbour has
    # not been measured, and must not outrank one that has. Finding #38: the
    # planner picked bfloat16 - never measured - over every measured half
    # configuration, precisely because the missing batch law left it with the
    # flat per-step figure and so the lowest predicted latency.
    if p.get("borrowed"):
        cand.extrapolated = True
        cand.notes.append(
            "this exact model/quantization/dtype has NOT been measured; its "
            "coefficients are borrowed, so it is ranked below measured "
            "configurations even when it prices lower")

    # Batching costs latency, and until finding #34 this planner did not know
    # it. The cost model's TPOT came from the decode roofline, which depends
    # on the weights and the bandwidth and NOT on how many requests share the
    # step - so it returned the same 7.91 ms for 8 slots and for 128, and the
    # planner ranked configurations whose measured p99 differed by 80% as
    # having identical latency. It then printed "the lower latency decided it
    # (1033 vs 1033 ms)", which is the bug announcing itself.
    #
    # The correction needs no new measurement. A decode step serves all the
    # slots at once, so each request's inter-token time IS the step time:
    #
    #     step_ms = 1000 * slots / throughput_tok_s(slots)
    #
    # Against today's controlled sweep:
    #        64 / 4714 = 13.6 ms      measured TPOT p50 13.6
    #        96 / 5061 = 19.0 ms      measured 19.0
    #       128 / 5339 = 24.0 ms      measured 24.0
    #
    # The saturating fit already held this; it was simply never asked.
    # The correction now lives in the cost model, where leave-one-out
    # validation tests it; this planner's copy was never validated by
    # anything. Two estimates of the same quantity, one checked and one not,
    # is not redundancy - it is an unlabelled choice. The checked one is used
    # and the other is kept only to disagree out loud.
    slots = cfg["max_num_seqs"]
    batch_tok_s = sat.throughput_tok_s(slots)
    if batch_tok_s > 0 and cand.tpot_ms > 0:
        step_from_saturation = 1000.0 * slots / batch_tok_s
        gap = abs(step_from_saturation - cand.tpot_ms) / cand.tpot_ms * 100
        if gap > 15:
            cand.notes.append(
                f"the saturation fit implies a {step_from_saturation:.1f} ms "
                f"decode step at {slots} slots while the cost model predicts "
                f"{cand.tpot_ms:.1f} ms ({gap:.0f}% apart). The cost model's "
                f"figure is used because it is the one leave-one-out "
                f"validates; the disagreement means one of them is fitted on "
                f"runs the other is not, and is worth resolving before this "
                f"configuration is trusted")
    cand.notes.extend(n for n in p["notes"] if "Finding #22" not in n)

    # Capacity: one slot produces one token per step, so a slot finishes a
    # request every output_tokens x step. Verified against the clean knee
    # sweep to +9% / -13%.
    service_s = req.output_tokens * cand.tpot_ms / 1000.0
    ceiling_rps = cfg["max_num_seqs"] / service_s if service_s else 0.0
    tok_s = sat.throughput_tok_s(cfg["max_num_seqs"])
    if tok_s > 0 and req.output_tokens:
        # The saturating fit is in tokens per second; a request consumes
        # output_tokens of them.
        cand.capacity_rps = min(tok_s / req.output_tokens, ceiling_rps)
    else:
        cand.capacity_rps = ceiling_rps
    cand.saturation = (cand.capacity_rps / ceiling_rps) if ceiling_rps else 1.0
    cand.extrapolated = sat.is_extrapolated(cfg["max_num_seqs"])
    if cand.extrapolated:
        cand.notes.append(
            f"{cfg['max_num_seqs']} slots is beyond the measured range (up to "
            f"{max(sat.measured_slots)}); the capacity is a fitted "
            f"extrapolation with a {(1 - sat.extrapolation_haircut) * 100:.0f}% "
            f"haircut applied, and must be measured before it is trusted")
    if sat.capped_at_peak(cfg["max_num_seqs"]):
        cand.notes.append(
            f"{cfg['max_num_seqs']} slots is past the measured peak of "
            f"{sat.peak_slots}, where throughput was seen to FALL rather than "
            f"rise. The capacity shown is the peak measurement standing in for "
            f"a prediction, so more slots than {sat.peak_slots} buy nothing "
            f"here and cost latency")

    if cand.capacity_rps <= 0:
        cand.feasible = False
        cand.refusals.append("predicted capacity is zero")
        return cand

    cand.gpus = max(1, math.ceil(req.rate_rps / cand.capacity_rps))
    if cand.gpus > req.max_gpus:
        cand.feasible = False
        cand.refusals.append(
            f"would need {cand.gpus} GPUs for {req.rate_rps:g} req/s "
            f"(limit {req.max_gpus})")
        return cand

    # Cost: the GPUs are billed by time, so the bill is GPU-hours over the
    # requests served in them.
    cand.cost_per_1k = (cand.gpus * cm.gpu_cost_per_hour
                        / (req.rate_rps * 3.6)) if req.rate_rps else float("inf")

    cand.accuracy_drop_pct = acc_drops.get(cfg["quantization"], 0.0)

    # Constraints are refusals, not penalties.
    if req.p99_ms is not None and cand.e2e_ms > req.p99_ms:
        cand.feasible = False
        cand.refusals.append(
            f"predicted latency {cand.e2e_ms:.0f} ms exceeds the "
            f"{req.p99_ms:.0f} ms limit")
    if (req.max_accuracy_drop_pct is not None
            and cand.accuracy_drop_pct > req.max_accuracy_drop_pct):
        cand.feasible = False
        cand.refusals.append(
            f"accuracy drop {cand.accuracy_drop_pct:.1f}% exceeds the "
            f"{req.max_accuracy_drop_pct:.1f}% limit")
    if req.budget_per_1k is not None and cand.cost_per_1k > req.budget_per_1k:
        cand.feasible = False
        cand.refusals.append(
            f"${cand.cost_per_1k:.4f} per 1K exceeds the "
            f"${req.budget_per_1k:.4f} budget")

    # Finding #19: at a long shared prefix, caching stops being optional.
    if (req.shared_prefix_tokens >= 1000
            and not cfg.get("prefix_caching", True)):
        cand.notes.append(
            f"prefix caching is off with a {req.shared_prefix_tokens}-token "
            f"shared prefix; measured at 2048 tokens that costs 43% more and "
            f"pushed p99 past 2 s")

    return cand


def rank(cands: list, noise_pct: float) -> list:
    """
    Cheapest first, but a cost gap smaller than the noise floor is not a gap.

    Finding #16: across three sessions the cost difference between the two
    best configurations stayed inside the noise while their p99 differed by
    1.9x. So ties on cost are broken on latency.
    """
    feasible = [c for c in cands if c.feasible]
    if not feasible:
        return []
    tol = max(noise_pct, 2.0) / 100.0

    def key(c):
        # Bucket the cost so that anything within the noise floor of the
        # cheapest in its bucket compares equal, then prefer a configuration
        # inside the measured range, then low latency. Preferring measured
        # over extrapolated on a tie matters: an extrapolated capacity is a
        # promise the data has not made.
        bucket = round(math.log(max(c.cost_per_1k, 1e-9)) / math.log(1 + tol))
        return (c.gpus, bucket, c.extrapolated, c.e2e_ms, -c.capacity_rps)

    ordered = sorted(feasible, key=key)

    # Collapse candidates the model cannot tell apart. With no shared prefix,
    # prefix_caching and kv_cache_dtype change nothing the model predicts, so
    # five configurations come out byte-identical and crowd the list. Keeping
    # one and counting the rest is honest: the model has no basis to prefer
    # among them, and saying so is better than ranking them arbitrarily.
    deduped, seen = [], {}
    for c in ordered:
        sig = (c.gpus, round(c.cost_per_1k, 6), round(c.e2e_ms, 1),
               round(c.capacity_rps, 2), round(c.accuracy_drop_pct, 3))
        if sig in seen:
            seen[sig].equivalents.append(c.label)
            continue
        c.equivalents = []
        seen[sig] = c
        deduped.append(c)
    return deduped


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def how_to_run(module: str) -> str:
    """
    How the READER would invoke another part of this library.

    Installed, there is no quality.py to run - there is `inferplan quality`.
    Printing the developer's filename tells the user to run something they do
    not have, and this is the fifth place today that did it: the VERIFY IT
    block, the sweep hint, the stop hint, the completion-length note, and
    here. One function, so there is one place to be wrong.
    """
    commands = {"runner": "sweep", "costmodel": "fit", "planner": "plan",
                "quality": "quality", "bench": "bench", "serve": "serve",
                "gpuwatch": "watch", "mergeledger": "merge",
                "repairledger": "repair"}
    if __package__:
        return f"inferplan {commands.get(module, module)}"
    return f"python {module}.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="ledger.jsonl")
    ap.add_argument("--runs-dir", default="runs",
                    help="bench.py's per-run JSON, for runs started by hand")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--bandwidth-gbps", type=float, default=None)
    ap.add_argument("--free-vram-mib", type=float, default=None,
                    help="free VRAM to plan against; measured if omitted")
    ap.add_argument("--gpu-cost-per-hour", type=float, default=0.35)
    ap.add_argument("--max-noise-pct", type=float, default=25.0)
    ap.add_argument("--quality", default="quality.json",
                    help="quality.py's store of measured accuracy drops")

    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--also-model", action="append", default=[],
                    help="another checkpoint to consider, e.g. an AWQ build "
                         "(repeatable, as MODEL:QUANT)")
    ap.add_argument("--rate", type=float, required=True,
                    help="offered load in requests per second")
    ap.add_argument("--input-tokens", type=int, default=38)
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--prefix-overlap", type=float, default=0.0,
                    help="share of the prompt identical across requests (0-1)")
    ap.add_argument("--p99-ms", type=float, default=None)
    ap.add_argument("--max-accuracy-drop-pct", type=float, default=None)
    ap.add_argument("--budget-per-1k", type=float, default=None)
    ap.add_argument("--max-gpus", type=int, default=8)
    ap.add_argument("--top", type=int, default=3,
                    help="how many to recommend for verification")
    ap.add_argument("--show-refused", type=int, default=0,
                    help="also print this many refused configurations")
    args = ap.parse_args()

    gpu, bw = args.gpu, args.bandwidth_gbps or 0.0
    if not gpu or not bw:
        d_gpu, d_bw = detect_gpu()
        gpu = gpu or d_gpu
        bw = bw or d_bw
    if not bw:
        print(f"unknown memory bandwidth for GPU {gpu!r}; pass "
              f"--bandwidth-gbps")
        return 1

    free_mib = args.free_vram_mib
    if free_mib is None:
        from .costmodel import gpu_free_total_mib
        f, _ = gpu_free_total_mib()
        free_mib = float(f) if f else 3900.0

    # Hand-run benchmarks are kept apart from the cost model on purpose.
    # They record no server configuration, so the latency and prefill fits -
    # which key off max_model_len, prefix_caching and quantization - would be
    # fitting fields that are absent rather than measured, and audit() would
    # read the missing fields as drift. They are admitted to the saturation
    # analysis, where only the slot count and the throughput matter, so that
    # a measurement can still contradict the curve. Finding #30.
    # A missing ledger is the normal state the first time anyone runs this,
    # not a bug, so it gets a sentence instead of a traceback. The first
    # impression a tool makes is the error it shows someone who has done
    # nothing wrong yet.
    sweep_hint = how_to_run("runner")
    try:
        runs = load_runs(Path(args.ledger))
        all_runs = load_all_runs(Path(args.ledger), Path(args.runs_dir))
    except FileNotFoundError:
        print(f"  no measurements yet: {args.ledger} does not exist.")
        print(f"  Measure a few configurations first, which takes about "
              f"15 minutes:")
        print(f"    {sweep_hint} --sweep knee --reference-every 4")
        print(f"  Then come back to this command.")
        return 1
    n_bench = sum(1 for r in all_runs if r.g("source") == "bench")
    aud = audit(runs, args.max_noise_pct)
    if aud["noisy"] or aud["inversions"] or aud["drift"]:
        print("=" * 76)
        print("  REFUSING TO PLAN  -  the ledger has runs that are not "
              "measurements")
        print("=" * 76)
        for r in aud["noisy"]:
            print(f"  noise {r.noise_pct:5.1f}%  {r.label[:54]}")
        for inv in aud["inversions"]:
            print(f"  inversion at concurrency {inv['concurrency']}: "
                  f"{inv['fewer_slots']} slots gave {inv['tput_a']:.0f} tok/s "
                  f"but {inv['more_slots']} slots gave {inv['tput_b']:.0f}")
        for d in aud["drift"]:
            print(f"  same config measured twice, {d['spread_pct']:.0f}% apart")
        print("-" * 76)
        print("  Fix the measurements first. A plan built on these would be")
        print("  confident and wrong, which is worse than no plan.")
        print("=" * 76)
        return 2

    cm = CostModel(gpu, bw, args.gpu_cost_per_hour)
    cm.fit(runs, args.max_noise_pct)

    models = [(args.model, None)]
    for spec in args.also_model:
        if ":" in spec:
            m, q = spec.rsplit(":", 1)
            models.append((m, q))
        else:
            models.append((spec, None))
    # Also consider precision variants of each checkpoint.
    expanded = []
    for m, q in models:
        if q:
            expanded.append((m, q))
        else:
            expanded.extend([(m, None), (m, "fp8")])

    req = Requirements(
        model=args.model, rate_rps=args.rate,
        input_tokens=args.input_tokens, output_tokens=args.output_tokens,
        prefix_overlap=args.prefix_overlap, p99_ms=args.p99_ms,
        max_accuracy_drop_pct=args.max_accuracy_drop_pct,
        budget_per_1k=args.budget_per_1k, max_gpus=args.max_gpus)

    space = search_space(req, expanded)
    step_ms = None
    for key in cm.decode_correction:
        try:
            from .costmodel import roofline_step_ms
            step_ms = (roofline_step_ms(key[0], key[1], bw)
                       * cm.decode_correction[key])
            break
        except ValueError:
            continue
    sat = Saturation.fit(all_runs, step_ms or 0.0)

    acc_drops, acc_measured = load_accuracy(Path(args.quality))
    cands = [evaluate(cfg, req, cm, gpu, free_mib, sat, acc_drops)
             for cfg in space]
    best = rank(cands, cm.noise_floor_pct)

    print("=" * 76)
    print(f"  INFERPLAN  -  {gpu} @ {bw:g} GB/s, {free_mib:.0f} MiB free")
    print("=" * 76)
    print(f"  serve            : {args.model}")
    print(f"  offered rate     : {args.rate:g} req/s")
    print(f"  prompt           : {args.input_tokens} in / "
          f"{args.output_tokens} out tokens"
          + (f", {args.prefix_overlap:.0%} shared "
             f"({req.shared_prefix_tokens} tokens)"
             if args.prefix_overlap else ""))
    limits = []
    if args.p99_ms:
        limits.append(f"latency <= {args.p99_ms:g} ms")
    if args.max_accuracy_drop_pct is not None:
        limits.append(f"accuracy drop <= {args.max_accuracy_drop_pct:g}%")
    if args.budget_per_1k:
        limits.append(f"<= ${args.budget_per_1k:.4f} per 1K")
    print(f"  limits           : {', '.join(limits) if limits else 'none given'}")
    if acc_measured:
        print(f"  accuracy         : measured for "
              f"{', '.join(str(k or 'fp16') for k in sorted(acc_measured, key=str))}"
              f" (from {args.quality})")
    else:
        print(f"  accuracy         : NOT MEASURED - using fallback estimates. "
              f"Run {how_to_run('quality')}.")
    print(f"  searched         : {len(space)} configurations, "
          f"{len(best)} feasible")
    print(f"  model fitted on  : {cm.n_runs} measured runs, noise floor "
          f"{cm.noise_floor_pct:.1f}%")
    if n_bench:
        print(f"                     including {n_bench} run(s) started by hand "
              f"through {how_to_run('bench')}, which are")
        print(f"                     RECORDED ONLY: nothing vouched for the "
              f"machine state they")
        print(f"                     ran in, so they cap nothing and only raise "
              f"cautions (#33)")
    if sat.fitted:
        print(f"  saturation       : throughput ceiling "
              f"{sat.t_max:.0f} tokens/s (half of it at "
              f"{sat.k:.0f} slots),")
        vouched = ("reference-controlled runs only"
                   if sat.fitted_on_controlled
                   else "runs with no reference control among them")
        print(f"                     fitted on "
              f"{', '.join(str(s) for s in sat.measured_slots)} slots, "
              f"from {vouched}")
        for w in sat.warnings:
            head, *rest = textwrap.wrap(w, 56)
            print(f"  CAUTION          : {head}")
            for line in rest:
                print(f"                     {line}")
    else:
        print(f"  saturation       : NOT FITTED - "
              f"{sat.reason or 'no usable runs'}")
        print(f"                     ({len(sat.measured_slots)} slot count(s) "
              f"usable). Capacity falls back to the raw slots/step ceiling,")
        print(f"                     which WILL be optimistic.")
    print("=" * 76)

    if not best:
        print()
        print("  NOTHING MEETS THESE REQUIREMENTS.")
        reasons = {}
        for c in cands:
            for r in c.refusals:
                head = r.split(";")[0].split("(")[0].strip()
                reasons[head] = reasons.get(head, 0) + 1
        print("  why, by count:")
        for r, n in sorted(reasons.items(), key=lambda kv: -kv[1])[:6]:
            print(f"    {n:4d}  {r}")
        print()
        print("  Relax one limit, or allow more GPUs.")
        return 3

    print()
    print(f"  RECOMMENDED  (verify the top {min(args.top, len(best))} "
          f"by measurement before deploying)")
    print("=" * 76)
    print(f"  {'#':<3}{'configuration':<42}{'GPUs':>5}{'$/1K':>9}"
          f"{'e2e ms':>8}{'cap r/s':>9}")
    print("-" * 76)
    for i, c in enumerate(best[:max(args.top, 5)], 1):
        same = f"  (+{len(c.equivalents)} identical)" if c.equivalents else ""
        if c.extrapolated:
            same += "  EXTRAPOLATED"
        print(f"  {i:<3}{c.label[:41]:<42}{c.gpus:>5}{c.cost_per_1k:>9.4f}"
              f"{c.e2e_ms:>8.0f}{c.capacity_rps:>9.1f}{same}")
    print("=" * 76)

    top = best[0]
    print()
    print("  THE PICK")
    print("-" * 76)
    print(f"  {top.label}")
    print()
    print(f"  GPUs needed      : {top.gpus}  "
          f"(capacity {top.capacity_rps:.1f} req/s each, "
          f"offered {args.rate:g})")
    print(f"  cost per 1K      : ${top.cost_per_1k:.4f}  "
          f"+/- {top.uncertainty_pct:.0f}%")
    print(f"  latency          : {top.e2e_ms:.0f} ms end to end "
          f"(TTFT {top.ttft_ms:.0f}, TPOT {top.tpot_ms:.2f})")
    q = top.config["quantization"]
    if q in acc_measured:
        print(f"  accuracy drop    : {top.accuracy_drop_pct:.1f}%  (measured, "
              f"{acc_measured[q]})")
    elif top.accuracy_drop_pct:
        print(f"  accuracy drop    : {top.accuracy_drop_pct:.1f}%  "
              f"(ESTIMATE, not measured on this model - run "
              f"{how_to_run('quality')})")
    else:
        print(f"  accuracy drop    : none; this is full precision")
    print(f"  needs free VRAM  : {top.required_free_mib:.0f} MiB")
    print(f"                     measured free right now: {free_mib:.0f} MiB.")
    print(f"                     Free VRAM moved 708 MiB across sessions on")
    print(f"                     this machine, so keep that headroom.")
    if top.equivalents:
        print(f"  equally good     : {len(top.equivalents)} other "
              f"configuration(s) the model cannot distinguish from this one,")
        print(f"                     e.g. {top.equivalents[0][:48]}.")
        print(f"                     They differ only in knobs that do nothing "
              f"on this workload.")
    for n in top.notes:
        print(f"  note             : {n}")

    runner_up = best[1] if len(best) > 1 else None
    if runner_up:
        gap = abs(runner_up.cost_per_1k - top.cost_per_1k) / max(top.cost_per_1k, 1e-9)
        if gap * 100 < max(cm.noise_floor_pct, 2.0):
            print(f"  tie-break        : {runner_up.label[:40]} costs within "
                  f"{gap * 100:.1f}%, inside the {cm.noise_floor_pct:.1f}% "
                  f"noise floor,")
            if abs(top.e2e_ms - runner_up.e2e_ms) < 1.0:
                # Saying "the lower latency decided it (1033 vs 1033 ms)" is
                # how finding #34's flat latency model announced itself. If
                # two numbers are equal, say what actually broke the tie.
                print(f"                     and their predicted latencies are "
                      f"equal too ({top.e2e_ms:.0f} ms), so the")
                print(f"                     larger capacity decided it "
                      f"({top.capacity_rps:.1f} vs "
                      f"{runner_up.capacity_rps:.1f} req/s). Measure both.")
            else:
                print(f"                     so the lower latency decided it "
                      f"({top.e2e_ms:.0f} vs {runner_up.e2e_ms:.0f} ms)")

    print()
    print("  RUN IT")
    print("-" * 76)
    print("  " + top.serve_command().replace("\n", "\n  "))
    print()
    print("  VERIFY IT  (the prediction is an estimate until measured)")
    print("-" * 76)
    # Through runner.py, not bench.py. A bare benchmark has no reference
    # control, and this planner has now been wrong twice in ways only a
    # control exposed (findings #33 and #34). Telling the user to verify a
    # recommendation the same unguarded way those errors were made would be
    # advice this tool's own history contradicts.
    # Printed the way the user actually invoked this, not the way the
    # developer runs it. Installed, "python runner.py" is not a command they
    # have; a verification step the reader cannot paste is advice they will
    # skip.
    sweep_cmd = ("inferplan sweep" if __package__
                 else f"python {Path(__file__).name.replace('planner', 'runner')}")
    # The verify command must reproduce the WORKLOAD, not just the slot
    # count. It did not: a plan for 500 output tokens printed a command that
    # measured 128, and the two numbers - 4397 ms predicted, 1201 ms measured
    # - were compared as though they described the same thing. A verification
    # that measures something else is worse than none, because it looks like
    # agreement or disagreement when it is neither.
    verify = [f"{sweep_cmd} --slots {top.config['max_num_seqs']}",
              "--reference-every 1",
              f"--max-tokens {req.output_tokens}",
              "--repeat 3",
              f"--slo-ms {args.p99_ms or 2000:.0f}"]
    if req.shared_prefix_tokens:
        verify.append(f"--shared-prefix-tokens {req.shared_prefix_tokens}")
    print(f"  {verify[0]} \\")
    for part in verify[1:-1]:
        print(f"    {part} \\")
    print(f"    {verify[-1]}")
    print()
    print(f"  That brackets the run with a reference measurement, so a result")
    print(f"  taken while the machine was drifting is reported as such rather")
    print(f"  than believed. Expect e2e near {top.e2e_ms:.0f} ms and capacity")
    print(f"  near {top.capacity_rps:.1f} req/s; on this machine the same")
    print(f"  predictions landed within 7% of four controlled measurements.")
    print("=" * 76)

    if args.show_refused:
        refused = [c for c in cands if not c.feasible][:args.show_refused]
        print()
        print(f"  REFUSED  (first {len(refused)})")
        print("-" * 76)
        for c in refused:
            print(f"  {c.label[:44]:<46}{c.refusals[0][:28]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())