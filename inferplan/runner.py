#!/usr/bin/env python3
"""
InferPlan runner: sweep orchestrator.

For each configuration in a sweep it will
  1. clamp gpu-memory-utilization to the ACTUAL free VRAM   (finding #1)
  2. start the vLLM server and time the startup              (finding #3)
  3. wait for /health, detecting early process death
  4. parse the server log for the backends really selected   (finding #2)
  5. run bench.py against it
  6. stop the server and wait for VRAM to drain
  7. append one ledger row, crash or not                     (finding #7, SLO-Guard)

Configurations are ordered so that runs sharing a torch.compile cache key run
consecutively, which avoids paying the ~28 s compile cost again and again.

Examples
    # see the plan without touching the GPU
    python runner.py --sweep baseline --dry-run

    # run it
    python runner.py --sweep baseline

    # resume: already-completed configs are skipped
    python runner.py --sweep baseline --resume
"""
import argparse
import itertools
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import urllib.error
import urllib.request

# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """One serving configuration: the knobs the planner will later search."""
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    quantization: Optional[str] = None      # None | "awq" | "gptq" | "fp8"
    dtype: str = "half"                     # half | bfloat16 | auto
    max_model_len: int = 2048
    max_num_seqs: int = 8                   # server-side batch limit
    gpu_memory_utilization: float = 0.70
    prefix_caching: bool = True             # vLLM default is ON (finding #5)
    chunked_prefill: bool = True            # vLLM default is ON (finding #5)
    kv_cache_dtype: str = "auto"            # auto | fp8

    # client-side load, not a server knob
    concurrency: int = 8
    requests: int = 40
    max_tokens: int = 128
    shared_prefix_tokens: int = 0   # identical preamble on every request
    # Appended to the label. The reference control re-runs a configuration a
    # sweep may also contain; without a tag the two would share a label and
    # the ledger's newest-wins rule would quietly replace a measurement with
    # a 128-request spot check.
    tag: str = ""

    def label(self) -> str:
        q = self.quantization or "fp16"
        bits = [q, self.dtype, f"b{self.max_num_seqs}", f"c{self.concurrency}",
                f"len{self.max_model_len}", f"u{self.gpu_memory_utilization:g}",
                f"kv{self.kv_cache_dtype}"]
        if self.shared_prefix_tokens:
            bits.append(f"sp{self.shared_prefix_tokens}")
        if not self.prefix_caching:
            bits.append("nopc")
        if not self.chunked_prefill:
            bits.append("nocp")
        if self.tag:
            bits.append(self.tag)
        return "_".join(bits)

    def compile_key(self) -> tuple:
        """
        Configs sharing this key reuse the same torch.compile artifacts, so
        running them back to back saves the compile step (finding #3).
        """
        # None is normalised to "" so keys stay sortable across configs where
        # quantization is unset.
        return (self.model, self.quantization or "", self.dtype,
                self.max_model_len, self.kv_cache_dtype)

    def serve_args(self, vllm_bin: str = "vllm") -> list:
        # The executable is a parameter because the library is often installed
        # in a different environment from vLLM itself - it measures whatever
        # server is running, so it should also be able to START a vLLM that
        # lives elsewhere. --serve-cmd cannot do this job: it replaces the
        # WHOLE command, flags included, so every config in a sweep would get
        # identical settings and the sweep would measure nothing.
        a = [vllm_bin, "serve", self.model,
             "--dtype", self.dtype,
             "--max-model-len", str(self.max_model_len),
             "--max-num-seqs", str(self.max_num_seqs),
             "--gpu-memory-utilization", f"{self.gpu_memory_utilization:g}",
             "--kv-cache-dtype", self.kv_cache_dtype]
        if self.quantization:
            a += ["--quantization", self.quantization]
        # Both are on by default in vLLM 0.33, so only the OFF case is passed.
        if not self.prefix_caching:
            a.append("--no-enable-prefix-caching")
        if not self.chunked_prefill:
            a.append("--no-enable-chunked-prefill")
        return a


# ----------------------------------------------------------------------------
# sweeps
# ----------------------------------------------------------------------------
def grid(**axes) -> list:
    """Cartesian product of named axes into Config objects."""
    names = list(axes)
    out = []
    for combo in itertools.product(*(axes[n] for n in names)):
        out.append(Config(**dict(zip(names, combo))))
    return out


SWEEPS = {
    # Concurrency sweep. Request count scales WITH concurrency so that every
    # config reaches a steady state: at 16 requests and concurrency 16 the
    # whole load is one batch and nothing steady is ever measured, which is
    # what made c16's first numbers unreliable (finding #10).
    "baseline": lambda: [Config(concurrency=c, requests=8 * c, max_num_seqs=8)
                         for c in (1, 4, 8, 16)],

    # Finding #8 hypothesis, tested on BOTH workload shapes, because the
    # answer depends on the workload and not on the flag.
    #   sp0    : no shared text, so there is no prefix to cache. Caching on
    #            vs off should show nothing - and that absence is the result,
    #            not a failed measurement.
    #   sp1024 : a long identical preamble, as production traffic has. Here
    #            caching should actually pay, and prefill is a larger share
    #            of the request, so the effect is visible.
    "prefix": lambda: [Config(prefix_caching=pc, shared_prefix_tokens=sp,
                              concurrency=8, requests=64, max_model_len=4096)
                       for sp in (0, 1024) for pc in (True, False)],

    # Finding #18 gave two points on the curve: 1.7% benefit at 0 shared
    # tokens, 29% at 1024. The planner needs to know WHERE the benefit
    # starts, so this fills in the points between and beyond.
    "prefix_curve": lambda: [Config(prefix_caching=pc, shared_prefix_tokens=sp,
                                    concurrency=8, requests=64,
                                    max_model_len=4096)
                             for sp in (128, 256, 512, 2048)
                             for pc in (True, False)],

    # Finding #7: does the saturation knee move with the server batch limit?
    "knee": lambda: [Config(max_num_seqs=m, concurrency=c, requests=8 * c)
                     for m in (8, 16, 32) for c in (m // 2, m, m * 2)],

    # Tests the planner's own extrapolation. Fitted on 8, 16 and 32 slots, the
    # saturation trend predicts 64 slots carry ~47 req/s rather than the 64 a
    # linear model promises. Nothing has been measured at 64, so the planner
    # flags that recommendation as EXTRAPOLATED. This sweep settles it.
    #
    # A prediction that is never checked against a measurement is just a
    # confident guess, which is the thing this whole project exists to avoid.
    # Finding #32 left this open: all three earlier attempts at 64-128 slots
    # were contaminated by platform drift. With --reference-every 1 each
    # config is bracketed by a reference reading, so a drift is caught
    # between configs rather than discovered a day later.
    "slots": lambda: [Config(max_num_seqs=s, concurrency=s, requests=8 * s)
                      for s in (64, 96, 128)],
    "knee64": lambda: [Config(max_num_seqs=64, concurrency=c, requests=8 * c)
                       for c in (32, 64)],

    # Memory-bound confirmed, so precision is the expected big lever.
    #
    # Two things that an earlier version of this sweep got wrong:
    #   - --quantization awq does NOT quantize an fp16 checkpoint on the fly.
    #     It needs weights that are already AWQ, so the MODEL changes too.
    #   - fp8 needs compute capability 8.9 (Ada/Hopper). RTX 3050 is 8.6, so
    #     it is expected to be refused here. It is kept in the sweep on
    #     purpose: a QUANT_UNSUPPORTED row is a real observation about this
    #     GPU, and the planner must learn not to offer fp8 on Ampere.
    "quant": lambda: [
        Config(model="Qwen/Qwen2.5-0.5B-Instruct", quantization=None,
               dtype="half", concurrency=8, requests=64),
        Config(model="Qwen/Qwen2.5-0.5B-Instruct", quantization=None,
               dtype="bfloat16", concurrency=8, requests=64),
        Config(model="Qwen/Qwen2.5-1.5B-Instruct-AWQ", quantization="awq",
               dtype="half", concurrency=8, requests=64),
        Config(model="Qwen/Qwen2.5-0.5B-Instruct", quantization="fp8",
               dtype="half", concurrency=8, requests=64),
    ],

    # Does an fp8 KV cache buy usable headroom on 3.18 GB?
    "kvcache": lambda: grid(kv_cache_dtype=["auto", "fp8"],
                            max_num_seqs=[8, 16], concurrency=[8],
                            requests=[64]),

    # A small smoke sweep for wiring checks.
    "smoke": lambda: grid(concurrency=[4], requests=[8], max_tokens=[32]),
}


# ----------------------------------------------------------------------------
# GPU helpers
# ----------------------------------------------------------------------------
def gpu_free_total_mib():
    """(free, total) VRAM in MiB via nvidia-smi, or (None, None)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True).stdout
        free, total = out.strip().splitlines()[0].split(",")
        return int(free.strip()), int(total.strip())
    except Exception:
        return None, None


def clamp_utilization(cfg: Config, headroom: float, reserve_mib: int,
                      log) -> Config:
    """
    Finding #1 and #12: declared VRAM is not usable VRAM, and what is free
    moves while the machine is in use.

    vLLM's --gpu-memory-utilization is a fraction of TOTAL memory, and at
    startup it refuses to run when free < util * total (that is exactly the
    error seen on 2026-11-05: util 0.85 wanted 3.4 of 4.0 GiB with 3.18 free).
    So the ceiling that passes that check is free/total.

    A fraction alone is not enough, though. The display reserve can GROW after
    the server starts, which would then hit a runtime OOM rather than a clean
    refusal. So a FIXED margin in MiB is held back as well, and the stricter
    of the two limits wins.
    """
    free, total = gpu_free_total_mib()
    if not free or not total:
        log("  no nvidia-smi reading; leaving gpu-memory-utilization as given")
        return cfg

    by_fraction = (free / total) * headroom          # passes vLLM's own check
    by_reserve = (free - reserve_mib) / total        # survives a growing reserve
    safe = max(min(by_fraction, by_reserve), 0.10)

    log(f"  VRAM free {free} / {total} MiB"
        f"  ->  ceiling {free / total:.3f},"
        f"  fraction {by_fraction:.3f},"
        f"  reserve-{reserve_mib}MiB {by_reserve:.3f}"
        f"  ->  safe {safe:.3f}")

    if cfg.gpu_memory_utilization <= safe:
        return cfg
    log(f"  clamping gpu-memory-utilization "
        f"{cfg.gpu_memory_utilization:g} -> {safe:.3f}")
    return replace(cfg, gpu_memory_utilization=round(safe, 3))


def wait_for_vram(threshold_mib: int, timeout: float, log):
    """After stopping a server, wait for its memory to actually be released."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        free, _ = gpu_free_total_mib()
        if free is None or free >= threshold_mib:
            return True
        time.sleep(2)
    log(f"  warning: VRAM did not return to {threshold_mib} MiB in {timeout:g}s")
    return False


# ----------------------------------------------------------------------------
# server lifecycle
# ----------------------------------------------------------------------------
HEALTH_TIMEOUT = 2.0

# What the server actually chose, as opposed to what we asked for (finding #2).
LOG_PATTERNS = {
    "attention_backend": re.compile(r"Using (\w+) attention backend"),
    "flash_attn_version": re.compile(r"Using FlashAttention version (\d+)"),
    "sampler_fallback": re.compile(r"FlashInfer top-p/top-k sampling unavailable"),
    "autotune_disabled": re.compile(r"Not enough SMs to use max_autotune"),
    "flashinfer_disabled": re.compile(r"FlashInfer kernels are disabled"),
    "pinned_memory_missing": re.compile(r"Pinned memory is not available"),
    "dtype_cast": re.compile(r"Casting torch\.(\w+) to torch\.(\w+)"),
    "model_load_s": re.compile(r"Model loading took [\d.]+ GiB memory and ([\d.]+) seconds"),
    "compile_s": re.compile(r"torch\.compile took ([\d.]+) s in total"),
    "kv_cache_gib": re.compile(r"GPU KV cache size: ([\d,]+) tokens"),
}

# Crash taxonomy: a failure is an observation, so it needs a label the cost
# model can use, not just a non-zero exit code.
CRASH_PATTERNS = [
    ("OOM_STARTUP", re.compile(r"Free memory on device .* is less than desired")),
    ("OOM_STARTUP", re.compile(r"No available memory for the cache blocks")),
    ("OOM_RUNTIME", re.compile(r"CUDA out of memory")),
    ("CTX_TOO_LONG", re.compile(r"larger than the maximum|max_model_len.*too large")),
    ("QUANT_UNSUPPORTED", re.compile(r"[Qq]uantization .* is not supported|"
                                     r"not supported on this GPU|"
                                     r"requires compute capability")),
    ("MODEL_NOT_FOUND", re.compile(r"Repository Not Found|does not appear to have a file")),
    ("ENGINE_INIT_FAILED", re.compile(r"Engine core initialization failed")),
]


def classify_crash(log_text: str) -> str:
    for name, pat in CRASH_PATTERNS:
        if pat.search(log_text):
            return name
    return "CRASH_OTHER"


def parse_server_log(path: Path) -> dict:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return {}
    facts = {}
    for key, pat in LOG_PATTERNS.items():
        m = pat.search(text)
        if not m:
            continue
        if m.groups():
            facts[key] = m.group(1) if len(m.groups()) == 1 else list(m.groups())
        else:
            facts[key] = True
    return facts


def health_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=HEALTH_TIMEOUT) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def served_model(url: str) -> Optional[str]:
    """The model id the server at `url` is actually serving, or None."""
    try:
        with urllib.request.urlopen(f"{url}/v1/models", timeout=HEALTH_TIMEOUT) as r:
            data = json.loads(r.read())
        entries = data.get("data") or []
        return entries[0].get("id") if entries else None
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None


def preflight_port(url: str, kill_stale: bool, log) -> bool:
    """
    Refuse to start while something else already answers on this URL.

    Without this check a leftover server from an earlier session answers
    /health instantly, the runner treats it as "ready in 0.0s", and every
    configuration in the sweep is benchmarked against THAT server's settings
    instead of its own. The numbers look fine and are entirely wrong.
    """
    if not health_ok(url):
        return True

    who = served_model(url) or "unknown model"
    log(f"  a server is ALREADY answering at {url} (serving {who})")

    if not kill_stale:
        log("  refusing to run: results would describe that server, not this")
        log("  config. Stop it first, or pass --kill-stale.")
        return False

    log("  --kill-stale given: stopping it")
    subprocess.run(["pkill", "-f", "vllm serve"], capture_output=True)
    deadline = time.time() + 60
    while time.time() < deadline:
        if not health_ok(url):
            log("  stale server gone")
            time.sleep(3)
            return True
        time.sleep(2)
    log("  stale server would not stop")
    return False


def start_server(cfg: Config, url: str, log_path: Path, timeout: float,
                 serve_cmd: Optional[str], log,
                 vllm_bin: str = "vllm"):
    """
    Launch the server and wait until it is healthy.

    Returns (proc, startup_seconds, reason). On success proc is the process and
    reason is None. On failure proc is None and reason distinguishes the two
    cases the cost model must not confuse: the process DIED (an error we can
    classify from its log) versus it never became healthy in time (a hang,
    which is a different observation).
    """
    args = (shlex.split(serve_cmd) if serve_cmd
            else cfg.serve_args(vllm_bin))
    log(f"  $ {' '.join(args)}")

    env = dict(os.environ)
    env.setdefault("VLLM_LOGGING_LEVEL", "INFO")

    with open(log_path, "w") as lf:
        proc = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)

    t0 = time.time()
    while True:
        if health_ok(url):
            elapsed = time.time() - t0
            # Confirm it is OUR server answering, serving OUR model.
            if proc.poll() is not None:
                log(f"  health is up but our process exited: another server "
                    f"holds {url}")
                return None, elapsed, "WRONG_SERVER"
            got = served_model(url)
            if got and got != cfg.model:
                log(f"  health is up but the server serves '{got}', "
                    f"not '{cfg.model}'")
                stop_server(proc, log)
                return None, elapsed, "WRONG_SERVER"
            log(f"  server ready in {elapsed:.1f}s (serving {got or cfg.model})")
            return proc, elapsed, None

        rc = proc.poll()
        if rc is not None:
            elapsed = time.time() - t0
            log(f"  server exited with code {rc} after {elapsed:.1f}s")
            if rc == 0:
                # Exit code 0 is not a crash: the command ran, did nothing,
                # and was content. In practice that means it was not a server
                # command at all - usually a --serve-cmd with the executable
                # but no arguments, which makes vllm print its usage and
                # leave. Calling that "DIED" sends the reader looking for an
                # out-of-memory error that never happened.
                log(f"  exit code 0 means it did not fail - it did nothing.")
                log(f"  That is what a server command with no arguments does.")
                log(f"  Check --serve-cmd, or use --vllm-bin to point at a")
                log(f"  vllm in another environment and keep the flags.")
                return None, elapsed, "NOT_A_SERVER"
            return None, elapsed, "DIED"

        if time.time() - t0 > timeout:
            log(f"  server did not become healthy in {timeout:g}s")
            stop_server(proc, log)
            return None, time.time() - t0, "STARTUP_TIMEOUT"

        time.sleep(2)


def stop_server(proc, log, grace: float = 25.0):
    """SIGTERM the process group, then SIGKILL whatever is left."""
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        proc.wait(timeout=grace)
        log("  server stopped")
        return
    except subprocess.TimeoutExpired:
        pass
    log("  server ignored SIGTERM; sending SIGKILL")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        log("  warning: server process would not die")


# ----------------------------------------------------------------------------
# benchmark invocation
# ----------------------------------------------------------------------------
def run_bench(cfg: Config, args, log, repeat: Optional[int] = None) -> dict:
    cmd = [sys.executable, args.bench, "--url", args.url,
           "--model", cfg.model,
           "--concurrency", str(cfg.concurrency),
           "--server-max-num-seqs", str(cfg.max_num_seqs),
           "--requests", str(cfg.requests),
           "--max-tokens", str(cfg.max_tokens),
           "--shared-prefix-tokens", str(cfg.shared_prefix_tokens),
           "--repeat", str(args.repeat if repeat is None else repeat),
           "--slo-ms", str(args.slo_ms),
           "--gpu-cost-per-hour", str(args.gpu_cost_per_hour),
           "--label", cfg.label(),
           "--csv", args.csv,
           "--json-dir", args.json_dir]
    log(f"  $ {' '.join(cmd[1:])}")
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=args.bench_timeout)
    dur = time.time() - t0
    if p.returncode != 0:
        log(f"  bench failed (rc={p.returncode})")
        log("  " + (p.stderr or p.stdout).strip().splitlines()[-1][:200])
        return {"bench_ok": False, "bench_seconds": dur,
                "bench_stderr": (p.stderr or "")[-1000:]}

    for line in p.stdout.splitlines():
        if any(k in line for k in ("pass ", "throughput", "cost per", "noise",
                                   "CONFIG", "INFEASIBLE")):
            log("  | " + line.strip())

    # The benchmark's own JSON holds the numbers; find the newest one.
    #
    # Matched EXACTLY, not by prefix. The old pattern was
    # glob(f"{label}_*.json"), and every label here is a prefix of longer
    # ones: "..._kvauto" is a prefix of "..._kvauto_nopc" and
    # "..._kvauto_sp512_nopc". So the glob for the plain config matched
    # another config's file, and because "s" sorts after "2" the
    # lexicographic "newest" was a sp512_nopc run rather than the run that
    # had just finished.
    #
    # The ledger therefore recorded, for every prefix-caching-ON run, the
    # metrics of a DIFFERENT configuration. Finding #40. The live output and
    # results.csv were correct the whole time, which is why it took a
    # ledger-versus-CSV comparison to see it.
    #
    # Only "<label>_<YYYYmmddTHHMMSSZ>.json" is accepted now.
    newest, record = None, {}
    d = Path(args.json_dir)
    if d.is_dir():
        exact = re.compile(
            rf"^{re.escape(cfg.label())}_\d{{8}}T\d{{6}}Z\.json$")
        files = sorted(f for f in d.glob(f"{cfg.label()}_*.json")
                       if exact.match(f.name))
        if files:
            newest = files[-1]
            try:
                record = json.loads(newest.read_text())
            except (OSError, json.JSONDecodeError):
                record = {}
    return {"bench_ok": True, "bench_seconds": dur,
            "bench_json": str(newest) if newest else None,
            "status": record.get("status"),
            "median": record.get("median"),
            "noise_pct": record.get("noise_pct")}


# ----------------------------------------------------------------------------
# ledger
# ----------------------------------------------------------------------------
def load_done(ledger: Path) -> set:
    done = set()
    if not ledger.exists():
        return done
    for line in ledger.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("label"):
            done.add(row["label"])
    return done


# ---------------------------------------------------------------------------
# reference control
# ---------------------------------------------------------------------------
# Finding #31/#32: one configuration measured 4730, 3083 and 801 tok/s across
# 25 hours on this machine, each measurement internally repeatable to 0.2%,
# and the cause was platform state rather than any knob. A sweep that does
# not re-measure a fixed configuration cannot tell a knob's effect from the
# machine's drift - and the drift here was 6x, larger than every knob in the
# search space put together.
#
# So a sweep now re-runs ONE small reference configuration at the start, every
# few configs, and at the end. The first reading is the baseline; every later
# one is compared to it. If the machine has moved, the sweep says so while it
# is still running, instead of a day later.
#
# The reference is deliberately small (128 requests, one pass, ~15 s) because
# its job is to detect a 6x move, not to measure anything to 1%.

def reference_config(model: str) -> Config:
    return Config(model=model, max_num_seqs=16, concurrency=16,
                  requests=128, max_tokens=128, tag="REF")


def run_reference(args, log, model: str) -> tuple:
    """
    Measure the reference configuration.

    Returns (tokens_per_second, error). An error is NOT the same as a missing
    number: it means the control could not be taken at all, and a sweep whose
    control cannot be taken is a sweep with no control. The first version of
    this returned None for both cases and the caller carried on with a
    cheerful "the control cannot speak this time" - which is precisely the
    failure findings #31 and #32 were about, reintroduced one layer up.
    """
    cfg = reference_config(model)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = Path(args.log_dir) / f"{cfg.label()}_{stamp}.log"
    if not preflight_port(args.url, args.kill_stale, log):
        return None, "another server holds the port"
    proc, startup_s, reason = start_server(cfg, args.url, log_path,
                                           args.startup_timeout,
                                           args.serve_cmd, log,
                                           args.vllm_bin)
    if proc is None:
        return None, f"the reference server would not start ({reason})"
    try:
        bench = run_bench(cfg, args, log, repeat=1)
    except subprocess.TimeoutExpired:
        bench = {}
        log("  reference bench timed out")
    stop_server(proc, log)
    # The main loop waits for VRAM and settles after every config; the
    # reference did not, and the very next preflight then found the
    # reference's own server still answering and aborted the sweep. Worse,
    # that same reference read 706 tok/s against an 1836 baseline measured
    # minutes earlier, because a leftover server was competing for the card -
    # so an unsettled port does not merely stall a sweep, it corrupts the one
    # measurement the sweep's trustworthiness rests on.
    for _ in range(45):
        if not health_ok(args.url):
            break
        time.sleep(2)
    else:
        log("  warning: the port is still answering after the reference run")
    time.sleep(3)

    # run_bench returns the benchmark's metrics nested under "median", not at
    # the top level. Reading the wrong key here made the control report "the
    # reference benchmark produced no throughput" immediately after printing
    # 1885.4 tokens/s on the line above - a failure mode worth naming, since
    # a guard that cries wolf gets switched off.
    tp = (bench.get("median") or {}).get("throughput_tok_s")
    try:
        tp = float(tp) if tp else None
    except (TypeError, ValueError):
        tp = None
    if tp is None:
        return None, "the reference benchmark produced no throughput"
    return tp, None


def reference_sanity(tok_s: float, free_before: Optional[float],
                     log) -> Optional[str]:
    """
    Refuse a reference reading that was taken on a card it did not have to
    itself. A baseline is the thing every later comparison is measured
    against, so a bad one is worse than none.
    """
    used, total = None, None
    try:
        free, total = gpu_free_total_mib()
    except Exception:
        return None
    if free is None or total is None:
        return None
    if free < total * 0.5:
        return (f"only {free:.0f} of {total:.0f} MiB were free after the "
                f"reference ran, so something else holds the card")
    return None


class ReferenceTrack:
    """
    Holds the reference readings and decides whether the sweep is still valid.

    The verdict is deliberately one-sided in its consequences but two-sided in
    its test: the machine getting FASTER mid-sweep invalidates a comparison
    just as thoroughly as it getting slower, because either way the configs
    measured before and after are not on the same machine.
    """

    def __init__(self, tolerance_pct: float):
        self.tolerance_pct = tolerance_pct
        self.readings = []          # (position, label, tok_s)
        self.baseline = None

    def add(self, position, tok_s) -> dict:
        if tok_s is None:
            # Kept tolerant, but check_reference no longer reaches here: a
            # reference that produced no number is handled as an error, not
            # as a reading, because a sweep must not continue unguarded.
            self.readings.append((position, None))
            return {"ok": True, "note": "reference unavailable"}
        self.readings.append((position, tok_s))
        if self.baseline is None:
            self.baseline = tok_s
            return {"ok": True, "baseline": tok_s}
        drift = (tok_s - self.baseline) / self.baseline * 100
        return {"ok": abs(drift) <= self.tolerance_pct,
                "drift_pct": drift, "baseline": self.baseline,
                "tok_s": tok_s}

    def drift_pct(self) -> float:
        """Drift of the latest usable reading against the baseline."""
        if self.baseline is None:
            return 0.0
        for _, tp in reversed(self.readings):
            if tp:
                return (tp - self.baseline) / self.baseline * 100
        return 0.0

    def latest(self) -> Optional[float]:
        for _, tp in reversed(self.readings):
            if tp:
                return tp
        return None

    def spread_pct(self) -> float:
        vals = [tp for _, tp in self.readings if tp]
        if len(vals) < 2:
            return 0.0
        return (max(vals) - min(vals)) / min(vals) * 100


def append_ledger(ledger: Path, row: dict):
    with open(ledger, "a") as f:
        f.write(json.dumps(row) + "\n")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="smoke", choices=sorted(SWEEPS),
                    help="which configuration set to run")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--bench", default=str(Path(__file__).with_name("bench.py")))
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--slo-ms", type=float, default=2000)
    ap.add_argument("--gpu-cost-per-hour", type=float, default=0.35)
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--json-dir", default="runs")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--ledger", default="ledger.jsonl")
    ap.add_argument("--startup-timeout", type=float, default=600)
    ap.add_argument("--bench-timeout", type=float, default=1800)
    ap.add_argument("--vram-headroom", type=float, default=0.90,
                    help="fraction of free VRAM to request at most")
    ap.add_argument("--vram-reserve-mib", type=int, default=350,
                    help="MiB of free VRAM held back for the display reserve, "
                         "which can grow while the server runs (finding #12)")
    ap.add_argument("--kill-stale", action="store_true",
                    help="stop a server already holding the URL instead of "
                         "refusing to run")
    ap.add_argument("--no-clamp", action="store_true",
                    help="do not adjust gpu-memory-utilization")
    ap.add_argument("--settle", type=float, default=5,
                    help="seconds to wait after stopping a server")
    ap.add_argument("--resume", action="store_true",
                    help="skip configs already in the ledger")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit; nothing is started")
    ap.add_argument("--vllm-bin", default="vllm",
                    help="path to the vllm executable, when it lives in a "
                         "different environment from this library, e.g. "
                         "~/myenv/bin/vllm. Unlike --serve-cmd this keeps "
                         "every config's own flags, so sweeps still work.")
    ap.add_argument("--serve-cmd", default=None,
                    help="replace the ENTIRE server command, flags included. "
                         "Every config then gets identical settings, so this "
                         "is for testing one server, not for sweeps. To point "
                         "at a vllm elsewhere, use --vllm-bin.")
    ap.add_argument("--limit", type=int, default=0,
                    help="run at most N configs")
    ap.add_argument("--slots", default=None,
                    help="comma-separated slot counts to measure, e.g. "
                         "32 or 64,96,128. Overrides --sweep. Exists so the "
                         "planner's own 'VERIFY IT' step can be run under a "
                         "reference control instead of by hand - findings #33 "
                         "and #34 both came from measurements that had none.")
    ap.add_argument("--reference-every", type=int, default=4,
                    help="re-measure the reference config every N configs "
                         "(0 turns the control off, which is not advised)")
    ap.add_argument("--reference-tolerance-pct", type=float, default=10.0,
                    help="how far the reference may move before the sweep is "
                         "called contaminated")
    ap.add_argument("--reference-rest-seconds", type=float, default=900,
                    help="rest for this long and re-check once when the "
                         "reference has moved; finding #32 measured ~15 min "
                         "as enough to recover fully")
    ap.add_argument("--ignore-reference", action="store_true",
                    help="keep going after the reference moves (the runs are "
                         "then marked contaminated in the ledger)")
    args = ap.parse_args()

    def log(msg=""):
        print(msg, flush=True)

    if args.slots:
        try:
            wanted = [int(s) for s in args.slots.replace(" ", "").split(",") if s]
        except ValueError:
            ap.error("--slots takes integers, e.g. --slots 32,64")
        if not wanted:
            ap.error("--slots was empty")
        configs = [Config(max_num_seqs=s, concurrency=s, requests=8 * s)
                   for s in wanted]
        args.sweep = "slots:" + ",".join(str(s) for s in wanted)
    else:
        configs = SWEEPS[args.sweep]()
    # Group by compile key so the compile cache is reused (finding #3).
    configs.sort(key=lambda c: (c.compile_key(), c.max_num_seqs, c.concurrency))

    # De-duplicate: several sweeps can generate the same config.
    seen, unique = set(), []
    for c in configs:
        if c.label() not in seen:
            seen.add(c.label())
            unique.append(c)
    configs = unique

    done = load_done(Path(args.ledger)) if args.resume else set()
    todo = [c for c in configs if c.label() not in done]
    if args.limit:
        todo = todo[:args.limit]

    log("=" * 70)
    log(f"  sweep '{args.sweep}': {len(configs)} config(s), "
        f"{len(todo)} to run, {len(configs) - len(todo)} skipped")
    log(f"  repeat {args.repeat}, SLO {args.slo_ms:g} ms, "
        f"ledger {args.ledger}")
    log("=" * 70)
    for i, c in enumerate(todo, 1):
        log(f"  {i:3d}. {c.label()}")
    log("=" * 70)

    if args.dry_run:
        groups = {}
        for c in todo:
            groups.setdefault(c.compile_key(), []).append(c)
        log(f"  {len(groups)} compile group(s); configs within a group reuse")
        log("  the torch.compile cache, saving roughly 28 s each after the first.")
        est = len(groups) * 60 + (len(todo) - len(groups)) * 25 \
            + len(todo) * args.repeat * 15
        log(f"  rough wall-clock estimate: {est / 60:.0f} min")
        log("  dry run: nothing started")
        return

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.json_dir).mkdir(parents=True, exist_ok=True)

    base_free, _ = gpu_free_total_mib()
    t_sweep = time.time()
    tally = {"OK": 0, "PARTIAL": 0, "FAILED": 0, "CRASH": 0}

    ref = ReferenceTrack(args.reference_tolerance_pct)
    contaminated = False
    control_failed = False
    # The reference must be served by the same checkpoint as the sweep,
    # or it would measure a different model's machine state.
    ref_model = (todo or configs)[0].model if (todo or configs) else \
        Config().model

    def check_reference(position: str) -> bool:
        """Measure the reference and report. False means stop the sweep."""
        nonlocal contaminated, control_failed
        if not args.reference_every:
            return True
        log()
        log("-" * 70)
        log(f"  REFERENCE CONTROL ({position})")
        log("-" * 70)
        tok_s, err = run_reference(args, log, ref_model)
        if not err and tok_s:
            bad = reference_sanity(tok_s, None, log)
            if bad:
                err = bad
        if err:
            control_failed = True
            log()
            log(f"  THE CONTROL COULD NOT BE TAKEN: {err}.")
            log("  Without it nothing says whether the machine moved during")
            log("  this sweep, which is the one thing the control exists for.")
            if args.ignore_reference:
                log("  --ignore-reference given: continuing unguarded, and")
                log("  every row will carry contaminated=true")
                contaminated = True
                return True
            stop_hint = ("inferplan serve --stop" if __package__
                         else "python serve.py --stop")
            log(f"  stopping. Fix it and start again:  {stop_hint}")
            contaminated = True
            return False
        verdict = ref.add(position, tok_s)
        append_ledger(Path(args.ledger), {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sweep": args.sweep, "kind": "reference", "position": position,
            "reference_tok_s": verdict.get("tok_s") or verdict.get("baseline"),
            "reference_baseline_tok_s": ref.baseline,
            "reference_drift_pct": verdict.get("drift_pct"),
            "outcome": "REFERENCE",
        })
        if "baseline" in verdict and "drift_pct" not in verdict:
            log(f"  baseline set: {ref.baseline:.0f} tok/s")
            return True
        d = verdict["drift_pct"]
        log(f"  reference now {verdict['tok_s']:.0f} tok/s against a baseline "
            f"of {ref.baseline:.0f} ({d:+.1f}%)")
        if verdict["ok"]:
            log(f"  within {args.reference_tolerance_pct:g}%; the machine has "
                f"not moved and the runs so far are comparable")
            return True

        log()
        log(f"  THE MACHINE HAS MOVED {d:+.1f}%, which is more than any knob")
        log(f"  in this sweep is expected to be worth. Every config measured")
        log(f"  since the last good reference is contaminated.")
        if args.reference_rest_seconds > 0:
            log(f"  resting {args.reference_rest_seconds / 60:.0f} min and "
                f"re-checking once (finding #32: 15 min restored 801 -> 4749)")
            time.sleep(args.reference_rest_seconds)
            retry_tp, retry_err = run_reference(args, log, ref_model)
            retry = ref.add(position + "+rest", retry_tp)
            if retry.get("ok") and "drift_pct" in retry:
                log(f"  recovered to {retry['tok_s']:.0f} tok/s "
                    f"({retry['drift_pct']:+.1f}%); carrying on")
                return True
            if "drift_pct" in retry:
                log(f"  still {retry['drift_pct']:+.1f}% after resting")
        contaminated = True
        if args.ignore_reference:
            log("  --ignore-reference given: continuing, and every remaining")
            log("  row will carry contaminated=true")
            return True
        log("  stopping the sweep. Hours of measurement that cannot be")
        log("  compared are worth less than none, because they get used.")
        return False

    if not check_reference("start"):
        log("=" * 70)
        log("  sweep abandoned at the first reference check")
        log("=" * 70)
        return 3

    for i, cfg in enumerate(todo, 1):
        label = cfg.label()
        log()
        log("-" * 70)
        log(f"[{i}/{len(todo)}] {label}")
        log("-" * 70)

        if not preflight_port(args.url, args.kill_stale, log):
            log("  aborting the sweep to avoid benchmarking the wrong server")
            break

        effective = cfg if args.no_clamp else clamp_utilization(
            cfg, args.vram_headroom, args.vram_reserve_mib, log)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = Path(args.log_dir) / f"{label}_{stamp}.log"

        # Finding #12: free VRAM moved 0.7 GB between two days with no config
        # change, so the reading at run time is part of the observation. A
        # recommendation is only portable to a machine with this much free.
        free_mib, total_mib = gpu_free_total_mib()

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sweep": args.sweep,
            "label": label,
            "config": asdict(cfg),
            "effective_config": asdict(effective),
            "vram_free_mib_at_start": free_mib,
            "vram_total_mib": total_mib,
            "server_log": str(log_path),
            # The machine state this run was measured in. Without it, two
            # rows in this ledger are not comparable and nothing says so.
            "reference_tok_s": ref.latest(),
            "reference_baseline_tok_s": ref.baseline,
            "reference_drift_pct": round(ref.drift_pct(), 2),
            "contaminated": contaminated,
        }

        proc, startup_s, reason = start_server(
            effective, args.url, log_path, args.startup_timeout,
            args.serve_cmd, log, args.vllm_bin)
        row["startup_seconds"] = round(startup_s, 2)

        if proc is None:
            if reason == "STARTUP_TIMEOUT":
                # The process is alive but never served: a hang, not an error.
                kind = "STARTUP_TIMEOUT"
            else:
                text = ""
                try:
                    text = log_path.read_text(errors="replace")
                except OSError:
                    pass
                kind = classify_crash(text)
            row.update({"outcome": "CRASH", "crash_kind": kind,
                        "server_facts": parse_server_log(log_path)})
            tally["CRASH"] += 1
            log(f"  CRASH: {kind} (kept as a negative observation)")
            append_ledger(Path(args.ledger), row)
            if base_free:
                wait_for_vram(int(base_free * 0.9), 60, log)
            time.sleep(args.settle)
            continue

        row["server_facts"] = parse_server_log(log_path)
        if row["server_facts"]:
            interesting = {k: v for k, v in row["server_facts"].items()
                           if k in ("attention_backend", "sampler_fallback",
                                    "autotune_disabled", "kv_cache_gib",
                                    "compile_s")}
            if interesting:
                log(f"  server chose: {interesting}")

        try:
            bench = run_bench(effective, args, log)
        except subprocess.TimeoutExpired:
            bench = {"bench_ok": False, "bench_stderr": "bench timeout"}
            log("  bench timed out")
        row.update(bench)

        # A server that died DURING the benchmark is a runtime crash, which is
        # a different observation from one that never started.
        if proc.poll() is not None:
            text = ""
            try:
                text = log_path.read_text(errors="replace")
            except OSError:
                pass
            row.update({"outcome": "CRASH", "crash_kind": classify_crash(text),
                        "crashed_during_bench": True})
            tally["CRASH"] += 1
            log(f"  server died during the benchmark: {row['crash_kind']}")
        else:
            status = row.get("status") or ("OK" if row.get("bench_ok") else "FAILED")
            row["outcome"] = status
            tally[status] = tally.get(status, 0) + 1

        # Re-read the log at the end; some facts are only printed later.
        row["server_facts"] = parse_server_log(log_path)
        append_ledger(Path(args.ledger), row)

        stop_server(proc, log)
        if base_free:
            wait_for_vram(int(base_free * 0.9), 60, log)
        time.sleep(args.settle)

        due = args.reference_every and (i % args.reference_every == 0) \
            and i < len(todo)
        if due and not check_reference(f"after {i}/{len(todo)}"):
            break

    if args.reference_every and todo:
        check_reference("end")

    mins = (time.time() - t_sweep) / 60
    log()
    log("=" * 70)
    log(f"  sweep '{args.sweep}' finished in {mins:.1f} min")
    log(f"  " + "  ".join(f"{k}={v}" for k, v in tally.items() if v))
    if args.reference_every:
        vals = [tp for _, tp in ref.readings if tp]
        if vals:
            log(f"  reference: {len(vals)} reading(s), "
                f"{min(vals):.0f}-{max(vals):.0f} tok/s, "
                f"spread {ref.spread_pct():.1f}%")
        if control_failed:
            # Saying "the reference moved by more than 10%" when the reference
            # could not be taken at all is a different fault reported as the
            # wrong one, and it sent the reader looking for a drift that never
            # happened.
            log()
            log("  THE CONTROL FAILED, so this sweep has no guarantee at all.")
            log("  The reference did not drift - it could not be measured.")
            log("  Nothing here should be used. Stop any running server")
            log("  (python serve.py --stop) and start again.")
        elif contaminated:
            log()
            log("  THIS SWEEP IS CONTAMINATED. The reference moved by more")
            log(f"  than {args.reference_tolerance_pct:g}% during it, so the")
            log("  configurations are not on a common machine state and must")
            log("  not be compared with each other. Re-run after a rest.")
        elif len(vals) > 1:
            log(f"  the machine held steady to {ref.spread_pct():.1f}%, so "
                f"these configs ARE comparable")
    else:
        log()
        log("  NO REFERENCE CONTROL was run (--reference-every 0). Findings")
        log("  #31 and #32 are what that costs: a 6x platform drift read as")
        log("  a property of the knob being turned.")
    log(f"  metrics -> {args.csv}")
    log(f"  ledger  -> {args.ledger}")
    log(f"  logs    -> {args.log_dir}/")
    log("=" * 70)


if __name__ == "__main__":
    main()