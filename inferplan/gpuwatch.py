#!/usr/bin/env python3
"""
InferPlan GPU watcher: find out why a measurement went wrong.

Run this in a second terminal while a sweep runs. It samples the GPU once a
second and, at the end, reports what moved. It exists because the knee sweep
of 23 Nov produced numbers that could not be configuration effects:

  - 16 batch slots served 2.4x LESS than 8 slots
  - one config's own passes disagreed by 80%
  - an identical config measured 32% lower than three days earlier

Any of those rules out the configuration as the cause, so the cause is in the
machine. The four candidates this script can tell apart:

  1. another process using the GPU      -> look at "other processes"
  2. thermal or power throttling        -> look at "throttle reasons" and clocks
  3. the GPU simply idle-waiting        -> low utilisation with high latency
                                           means the bottleneck is elsewhere
  4. the client or host starved of RAM  -> host memory and swap

Usage
    python gpuwatch.py                       # until Ctrl+C, then a report
    python gpuwatch.py --seconds 120         # fixed window
    python gpuwatch.py --csv watch.csv       # also save every sample
"""
import argparse
import csv
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime

FIELDS = [
    "timestamp", "utilization.gpu", "utilization.memory",
    "memory.used", "memory.total", "temperature.gpu",
    "power.draw", "clocks.current.sm", "clocks.max.sm",
    "clocks_throttle_reasons.active",
]

# nvidia-smi reports throttling as a bitmask; these are the bits that matter.
THROTTLE_BITS = {
    0x0000000001: "gpu_idle",
    0x0000000002: "applications_clocks_setting",
    0x0000000004: "sw_power_cap",
    0x0000000008: "hw_slowdown",
    0x0000000010: "sync_boost",
    0x0000000020: "sw_thermal_slowdown",
    0x0000000040: "hw_thermal_slowdown",
    0x0000000080: "hw_power_brake_slowdown",
    0x0000000100: "display_clock_setting",
}
# Only these mean the GPU was actually held back.
#
# sw_power_cap and sw_thermal_slowdown are NOT in this set, though they sound
# like they belong. Finding #32: on this laptop GPU both were reported active
# in 201 of 209 samples across a 231-second run, while the driver's own
# counters totalled 73 ms for each - and byte-identical totals for two
# supposedly distinct causes (73445 us and 73445 us) is one combined flag,
# not two measurements. Worse, the flag pointed the wrong way: the run with
# those bits set at a 1410 MHz clock produced 1051 tok/s while a run at
# 945 MHz produced 3083. A signal that is asserted almost always, and
# anti-correlates with slowness when it varies, is not evidence of throttling.
#
# The clock while busy and the power draw carried the real information in that
# episode, and both are reported below regardless of any flag.
REAL_THROTTLES = {"hw_slowdown", "hw_thermal_slowdown",
                  "hw_power_brake_slowdown"}
# Reported, but as context rather than as a verdict.
SOFT_FLAGS = {"sw_power_cap", "sw_thermal_slowdown"}


def decode_throttle(raw: str) -> list:
    raw = (raw or "").strip()
    if not raw or raw in ("N/A", "[N/A]", "Not Active"):
        return []
    try:
        mask = int(raw, 16) if raw.lower().startswith("0x") else int(raw)
    except ValueError:
        return [raw]
    return [name for bit, name in THROTTLE_BITS.items() if mask & bit]


def sample_gpu() -> dict:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(FIELDS[1:])}",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20, check=True).stdout
    parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
    row = dict(zip(FIELDS[1:], parts))
    row["timestamp"] = datetime.now().strftime("%H:%M:%S")
    return row


def sample_processes() -> list:
    """Every process holding GPU memory, including ones we did not start."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True).stdout
    except Exception:
        return []
    procs = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        bits = [b.strip() for b in line.split(",")]
        if len(bits) >= 3:
            procs.append({"pid": bits[0], "name": bits[1], "mib": bits[2]})
    return procs


def sample_host() -> dict:
    """Host memory and swap, because a starved client looks like a slow GPU."""
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                    info[k] = int(v.split()[0]) / 1024.0      # MiB
    except OSError:
        pass
    try:
        with open("/proc/loadavg") as f:
            info["load1"] = float(f.read().split()[0])
    except OSError:
        pass
    return info


def num(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0,
                    help="sample for this long; 0 means until Ctrl+C")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--csv", default=None, help="also write every sample here")
    args = ap.parse_args()

    if not shutil.which("nvidia-smi"):
        print("nvidia-smi not found; nothing to watch")
        return 1

    samples, procs_seen, throttles = [], {}, {}
    stop = {"now": False}

    def on_sigint(_sig, _frm):
        stop["now"] = True
    signal.signal(signal.SIGINT, on_sigint)

    print("sampling the GPU once a second; Ctrl+C for the report")
    print(f"{'time':>9}{'util%':>7}{'mem MiB':>9}{'temp C':>8}"
          f"{'watt':>7}{'SM MHz':>8}  throttle / other processes")
    print("-" * 78)

    t0 = time.time()
    writer = csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS + ["other_procs"])
        writer.writeheader()

    try:
        while not stop["now"]:
            if args.seconds and time.time() - t0 > args.seconds:
                break
            try:
                row = sample_gpu()
            except Exception as e:
                print(f"  sample failed: {type(e).__name__}")
                time.sleep(args.interval)
                continue

            procs = sample_processes()
            for p in procs:
                procs_seen.setdefault(p["name"], set()).add(p["pid"])
            active = decode_throttle(row.get("clocks_throttle_reasons.active"))
            for a in active:
                throttles[a] = throttles.get(a, 0) + 1

            row["host"] = sample_host()
            samples.append(row)

            flags = [a for a in active if a in REAL_THROTTLES]
            others = [p["name"].split("/")[-1] for p in procs]
            tail = " ".join(flags) + ("   procs: " + ", ".join(others)
                                      if others else "")
            print(f"{row['timestamp']:>9}{num(row['utilization.gpu']):>7.0f}"
                  f"{num(row['memory.used']):>9.0f}"
                  f"{num(row['temperature.gpu']):>8.0f}"
                  f"{num(row['power.draw']):>7.0f}"
                  f"{num(row['clocks.current.sm']):>8.0f}  {tail}")

            if writer:
                out = {k: row.get(k, "") for k in FIELDS}
                out["other_procs"] = "|".join(others)
                writer.writerow(out)
                csv_file.flush()

            time.sleep(args.interval)
    finally:
        if csv_file:
            csv_file.close()

    if not samples:
        print("no samples taken")
        return 1

    utils = [num(s["utilization.gpu"]) for s in samples]
    temps = [num(s["temperature.gpu"]) for s in samples]
    powers = [num(s["power.draw"]) for s in samples]
    clocks = [num(s["clocks.current.sm"]) for s in samples]
    mems = [num(s["memory.used"]) for s in samples]
    max_clock = max((num(s["clocks.max.sm"]) for s in samples), default=0)

    print()
    print("=" * 78)
    print(f"  REPORT  -  {len(samples)} samples over "
          f"{time.time() - t0:.0f} s")
    print("=" * 78)
    print(f"  GPU utilisation : {min(utils):5.0f} - {max(utils):5.0f} %   "
          f"median {statistics.median(utils):5.0f} %")
    print(f"  memory used     : {min(mems):5.0f} - {max(mems):5.0f} MiB")
    print(f"  temperature     : {min(temps):5.0f} - {max(temps):5.0f} C")
    print(f"  power draw      : {min(powers):5.0f} - {max(powers):5.0f} W")
    print(f"  SM clock        : {min(clocks):5.0f} - {max(clocks):5.0f} MHz"
          + (f"   (max {max_clock:.0f})" if max_clock else ""))
    hosts = [s["host"] for s in samples if s.get("host")]
    if hosts:
        avail = [h.get("MemAvailable", 0) for h in hosts]
        swapfree = [h.get("SwapFree", 0) for h in hosts]
        swaptotal = hosts[0].get("SwapTotal", 0)
        loads = [h.get("load1", 0) for h in hosts]
        print(f"  host RAM free    : {min(avail):5.0f} - {max(avail):5.0f} MiB")
        if swaptotal:
            used = [swaptotal - s for s in swapfree]
            print(f"  swap in use      : {min(used):5.0f} - {max(used):5.0f} MiB"
                  f"   of {swaptotal:.0f}")
        print(f"  load average     : {min(loads):5.2f} - {max(loads):5.2f}")
    print("-" * 78)

    verdicts = []

    real = {k: v for k, v in throttles.items() if k in REAL_THROTTLES}
    if real:
        total = len(samples)
        for name, n in sorted(real.items(), key=lambda kv: -kv[1]):
            verdicts.append(
                f"THROTTLING: {name} active in {n}/{total} samples "
                f"({n / total * 100:.0f}% of the time)")
    else:
        verdicts.append("no HARDWARE power or thermal throttling was reported")

    soft = {k: v for k, v in throttles.items() if k in SOFT_FLAGS}
    if soft:
        total = len(samples)
        names = ", ".join(f"{k} {v}/{total}" for k, v in sorted(soft.items()))
        verdicts.append(
            f"soft flags (context only, NOT a throttling verdict): {names}. "
            f"On this hardware these are asserted under almost any load and "
            f"anti-correlated with slowness when they varied - finding #32. "
            f"Judge by the busy clock and power draw below.")

    # A low clock only means throttling if the GPU was actually being asked to
    # work. An idle GPU downclocks to ~200 MHz by design, and reporting that as
    # a 90% cut is a false alarm.
    busy = [(c, u) for c, u in zip(clocks, utils) if u > 20]
    # Always reported, not only when something looks wrong. Finding #32: the
    # busy clock and the power draw were the two numbers that distinguished a
    # fast run from a slow one, and a run with 3x less throughput drew LESS
    # power - the signature of a GPU waiting for work rather than one being
    # held back. Comparing these across runs is the point, so they must be
    # printed every time, not conditionally.
    if busy:
        bc = sorted(c for c, _ in busy)
        verdicts.append(
            f"BUSY CLOCK median {bc[len(bc) // 2]:.0f} MHz over {len(busy)} "
            f"busy sample(s), range {bc[0]:.0f}-{bc[-1]:.0f} MHz")
    busy_power = [p for p, u in zip(powers, utils) if u > 20] if powers else []
    if busy_power:
        bp = sorted(busy_power)
        verdicts.append(
            f"BUSY POWER  median {bp[len(bp) // 2]:.1f} W, range "
            f"{bp[0]:.1f}-{bp[-1]:.1f} W. Less power at a similar clock than "
            f"another run means the GPU was idle waiting, not throttled.")
    if max_clock and busy:
        low = min(c for c, _ in busy)
        if low < max_clock * 0.7:
            verdicts.append(
                f"CLOCKS DROPPED to {low:.0f} MHz while the GPU was busy, "
                f"against a maximum of {max_clock:.0f} MHz "
                f"({(1 - low / max_clock) * 100:.0f}% cut)")
    elif max_clock and not busy:
        verdicts.append(
            f"the GPU was never busy during this window, so the low clock "
            f"({min(clocks):.0f} of {max_clock:.0f} MHz) is just idle power "
            f"saving, not throttling")

    # Memory held while nothing is running is the signature of an orphan
    # server, and that is what starves the next run's KV cache.
    if utils and max(utils) < 10 and mems and min(mems) > 300:
        verdicts.append(
            f"MEMORY HELD WHILE IDLE: {min(mems):.0f} MiB occupied with the "
            f"GPU at {max(utils):.0f}% busy. Something is sitting on the GPU "
            f"without using it, most likely a server left over from an "
            f"earlier run. The next run would start with only "
            f"{max((num(s['memory.total']) for s in samples), default=0) - min(mems):.0f} "
            f"MiB free and get a much smaller KV cache.")

    foreign = {n: p for n, p in procs_seen.items()
               if "vllm" not in n.lower() and "python" not in n.lower()}
    if foreign:
        verdicts.append("OTHER GPU PROCESSES were present: "
                        + ", ".join(sorted(foreign)))
    elif len(procs_seen) > 1:
        verdicts.append("more than one python/vllm process held GPU memory: "
                        + ", ".join(sorted(procs_seen)))

    if utils and statistics.median(utils) < 50:
        verdicts.append(
            f"GPU was only {statistics.median(utils):.0f}% busy at the median. "
            f"If latency was high anyway, the bottleneck was NOT the GPU "
            f"(look at the client, the host, or queueing)")

    if hosts:
        if min(h.get("MemAvailable", 1e9) for h in hosts) < 800:
            verdicts.append(
                "HOST RAM fell below 800 MiB free; the benchmark client itself "
                "may have been starved, which looks like a slow server")
        st = hosts[0].get("SwapTotal", 0)
        if st:
            used_max = max(st - h.get("SwapFree", st) for h in hosts)
            if used_max > 0.25 * st:
                verdicts.append(
                    f"SWAP in use up to {used_max:.0f} MiB; host memory "
                    f"pressure can slow the client")

    for v in verdicts:
        print(f"  {v}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())