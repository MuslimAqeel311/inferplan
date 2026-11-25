#!/usr/bin/env python3
"""
InferPlan serve: start exactly one vLLM server, and wait until it answers.

This exists because starting servers by hand has gone wrong twice on this
project, both times silently:

  1. An orphan server left holding 2919 MiB made a whole knee sweep read as
     if 16 batch slots were 2.4x WORSE than 8. The numbers had 0.1% internal
     noise and were wrong by 7x.
  2. Three servers were started at once on a 4 GiB card. They competed for
     memory, all three died, and no measurement came out at all.

`vllm serve ... &` offers no protection against either: it returns
immediately, so the shell prompt comes back long before the server is ready,
and nothing stops a second one starting next to the first.

This script refuses to start while another server holds the port, kills the
old one when asked, waits for VRAM to actually drain, and blocks until
/health answers - so the next command can assume a working server.

Usage
    python serve.py Qwen/Qwen2.5-0.5B-Instruct
    python serve.py Qwen/Qwen2.5-1.5B-Instruct-AWQ --quantization awq
    python serve.py --stop
"""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def health_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def served_model(url: str) -> str:
    try:
        with urllib.request.urlopen(f"{url}/v1/models", timeout=2) as r:
            data = json.loads(r.read())
        return (data.get("data") or [{}])[0].get("id", "")
    except Exception:
        return ""


def gpu_used_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True).stdout
        used, total = out.strip().splitlines()[0].split(",")
        return int(used.strip()), int(total.strip())
    except Exception:
        return None, None


def stop_all(url: str, quiet: bool = False) -> bool:
    """Stop every vLLM server and wait for the GPU memory to come back."""
    running = health_ok(url)
    if running and not quiet:
        print(f"  stopping the server at {url} "
              f"(serving {served_model(url) or 'unknown'})")
    subprocess.run(["pkill", "-f", "vllm serve"], capture_output=True)
    subprocess.run(["pkill", "-f", "VLLM::EngineCore"], capture_output=True)

    # Waiting for the port to close is not enough: the engine process can
    # outlive the API server and keep holding VRAM, which is exactly how the
    # orphan in finding #20 survived.
    deadline = time.time() + 90
    while time.time() < deadline:
        used, total = gpu_used_mib()
        port_free = not health_ok(url, timeout=1.0)
        if port_free and (used is None or used < 300):
            if not quiet:
                print(f"  stopped; GPU holds {used if used is not None else '?'} MiB")
            return True
        time.sleep(2)
    used, _ = gpu_used_mib()
    print(f"  WARNING: after 90 s the GPU still holds {used} MiB and/or the "
          f"port is still answering.")
    print(f"  Check with: nvidia-smi --query-compute-apps=pid,process_name,"
          f"used_memory --format=csv")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", help="model to serve")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--dtype", default="half")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument("--vllm-bin", default="vllm",
                    help="path to the vllm executable, for when it lives in "
                         "a different environment from this library")
    ap.add_argument("--log", default="server.log")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--stop", action="store_true",
                    help="stop any running server and exit")
    ap.add_argument("--extra", default="",
                    help="extra flags passed through to vllm serve")
    args = ap.parse_args()

    if args.stop:
        print("stopping any running vLLM server")
        return 0 if stop_all(args.url) else 1

    if not args.model:
        ap.error("a model is required unless --stop is given")

    print("=" * 70)
    print(f"  serving {args.model}")
    print("=" * 70)

    # One server at a time. Always.
    if health_ok(args.url) or (gpu_used_mib()[0] or 0) > 300:
        if not stop_all(args.url):
            print("  refusing to start a second server next to the old one.")
            return 1
    else:
        print("  GPU is clear")

    cmd = [args.vllm_bin, "serve", args.model,
           "--dtype", args.dtype,
           "--max-model-len", str(args.max_model_len),
           "--max-num-seqs", str(args.max_num_seqs),
           "--gpu-memory-utilization", f"{args.gpu_memory_utilization:g}",
           "--kv-cache-dtype", args.kv_cache_dtype]
    if args.quantization:
        cmd += ["--quantization", args.quantization]
    if args.no_prefix_caching:
        cmd.append("--no-enable-prefix-caching")
    if args.extra:
        cmd += shlex.split(args.extra)

    log_path = Path(args.log)
    print(f"  $ {' '.join(cmd)}")
    print(f"  log -> {log_path}")

    try:
        with open(log_path, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    start_new_session=True)
    except FileNotFoundError:
        print()
        print(f"  {args.vllm_bin!r} was not found.")
        print("  Activate the environment that has vLLM, or pass")
        print("  --vllm-bin /path/to/venv/bin/vllm to use one elsewhere.")
        return 4
    except OSError as e:
        print(f"  could not start the server: {type(e).__name__}: {e}")
        return 4

    t0 = time.time()
    last = 0.0
    while True:
        if health_ok(args.url):
            got = served_model(args.url)
            used, total = gpu_used_mib()
            print(f"  ready in {time.time() - t0:.0f}s, serving {got}")
            if used is not None:
                print(f"  GPU holds {used} of {total} MiB")
            print("=" * 70)
            print(f"  stop it with: python serve.py --stop")
            return 0

        rc = proc.poll()
        if rc is not None:
            print(f"  the server exited with code {rc} after "
                  f"{time.time() - t0:.0f}s")
            print(f"  last lines of {log_path}:")
            try:
                for line in log_path.read_text(errors="replace").splitlines()[-12:]:
                    print(f"    {line[:160]}")
            except OSError:
                pass
            return 2

        if time.time() - t0 > args.timeout:
            print(f"  not ready after {args.timeout:g}s; stopping it")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()
            return 3

        if time.time() - last > 15:
            print(f"  waiting... {time.time() - t0:.0f}s")
            last = time.time()
        time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())