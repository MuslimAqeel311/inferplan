#!/usr/bin/env python3
"""
InferPlan benchmark engine (v2).

Measures one serving configuration against a running vLLM server and writes
one row per repeat to a CSV, plus a JSON record with full metadata.

Fixes over v1:
  1. Real token counts from vLLM's usage block (not SSE chunk counts).
  2. Client concurrency and server max-num-seqs recorded separately; queue
     delay reported apart from service time.
  3. Per-request error handling: failures are counted, never silently folded
     into the latency distribution, and never abort the run.
  4. --repeat with median + spread (measurement noise is ~7% on this setup).
  5. True ITL distribution (per inter-token gap), not just averaged TPOT.
  6. Goodput: share of requests that met the SLO.

Usage:
    python bench.py --concurrency 8 --requests 40 --repeat 3 \
        --label fp16_c8 --server-max-num-seqs 8 --slo-ms 2000
"""
import argparse
import asyncio
import json
import os
import statistics
import time
from datetime import datetime, timezone

import aiohttp

DEFAULT_URL = "http://localhost:8000"

PROMPTS = [
    "Explain GPU memory bandwidth in simple terms.",
    "What is the difference between prefill and decode in LLM inference?",
    "Write a short summary of why quantization reduces memory use.",
    "Describe what a KV cache stores and why it grows.",
    "How does continuous batching improve GPU utilization?",
    "What does p99 latency mean for a web service?",
    "Explain the roofline model in two sentences.",
    "Why is inference usually more expensive than training over time?",
]

# A long shared preamble, used only when --shared-prefix-tokens is set.
#
# Why this exists: the prompts above have no text in common, so a run against
# them cannot show what prefix caching does - there is no prefix to re-use, and
# measuring "caching on vs off" on that workload would show no difference for
# the wrong reason. Production traffic usually DOES repeat a long system prompt
# on every request, so that case needs its own workload.
_PREAMBLE_UNIT = (
    "You are a careful technical assistant for a platform engineering team. "
    "Answer precisely, prefer measured numbers over estimates, state your "
    "assumptions, and keep responses short. Never invent benchmark figures. "
    "When a question concerns serving performance, distinguish prefill from "
    "decode, and latency from throughput. "
)


def shared_prefix(approx_tokens: int) -> str:
    """~4 characters per token is close enough for a load-generator preamble."""
    if approx_tokens <= 0:
        return ""
    target_chars = approx_tokens * 4
    reps = max(1, target_chars // len(_PREAMBLE_UNIT) + 1)
    return (_PREAMBLE_UNIT * reps)[:target_chars]


# ----------------------------------------------------------------------------
# one request
# ----------------------------------------------------------------------------
async def one_request(session, url, model, prompt, max_tokens, req_id):
    """
    Send one streaming request.

    Returns a dict that ALWAYS has an "ok" key. On failure, ok=False and the
    reason is recorded; the caller counts it instead of treating it as latency.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
        # vLLM returns a final chunk carrying real token counts.
        "stream_options": {"include_usage": True},
    }

    start = time.perf_counter()
    ttft = None
    last_tok_at = None
    itls = []            # inter-token latencies, one per gap
    chunk_count = 0      # SSE content chunks (NOT tokens)
    usage = None

    try:
        async with session.post(f"{url}/v1/chat/completions", json=payload) as resp:
            if resp.status != 200:
                body = (await resp.text())[:300]
                return {"ok": False, "req_id": req_id,
                        "error": f"HTTP {resp.status}", "detail": body}

            async for raw in resp.content:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue

                # With include_usage the final chunk has usage and may carry
                # an empty choices list - handle both shapes.
                if chunk.get("usage"):
                    usage = chunk["usage"]

                choices = chunk.get("choices") or []
                if not choices:
                    continue

                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - start
                    else:
                        itls.append((now - last_tok_at) * 1000)
                    last_tok_at = now
                    chunk_count += 1

    except asyncio.TimeoutError:
        return {"ok": False, "req_id": req_id, "error": "timeout"}
    except aiohttp.ClientError as e:
        return {"ok": False, "req_id": req_id,
                "error": type(e).__name__, "detail": str(e)[:200]}

    end = time.perf_counter()

    if ttft is None:
        return {"ok": False, "req_id": req_id, "error": "no_tokens_returned"}

    # Prefer the server's own token counts; fall back to chunk count and say so.
    if usage:
        out_tokens = usage.get("completion_tokens", 0)
        in_tokens = usage.get("prompt_tokens", 0)
        token_source = "usage"
    else:
        out_tokens = chunk_count
        in_tokens = 0
        token_source = "chunk_count_fallback"

    decode_s = (end - start) - ttft
    return {
        "ok": True,
        "req_id": req_id,
        "ttft_ms": ttft * 1000,
        "e2e_ms": (end - start) * 1000,
        "decode_ms": decode_s * 1000,
        "out_tokens": out_tokens,
        "in_tokens": in_tokens,
        "chunk_count": chunk_count,
        # TPOT from REAL token count, excluding the first token
        "tpot_ms": decode_s * 1000 / max(out_tokens - 1, 1),
        "itls": itls,
        "token_source": token_source,
    }


# ----------------------------------------------------------------------------
# one repeat of the whole load
# ----------------------------------------------------------------------------
async def one_pass(url, model, concurrency, total, max_tokens, prefix=""):
    sem = asyncio.Semaphore(concurrency)

    async def guarded(session, i):
        async with sem:
            prompt = prefix + PROMPTS[i % len(PROMPTS)]
            return await one_request(
                session, url, model, prompt, max_tokens, i)

    timeout = aiohttp.ClientTimeout(total=900, sock_read=300)
    conn = aiohttp.TCPConnector(limit=max(concurrency * 2, 32))
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        # Warm-up, not measured. It uses the same prefix so that a cached
        # prefix is already warm when the measured requests start: otherwise
        # the first measured request pays for filling the cache and the
        # comparison between caching on and off is skewed by one request.
        await one_request(session, url, model, prefix + PROMPTS[0], 16, -1)

        wall_start = time.perf_counter()
        # return_exceptions keeps one bad request from aborting the whole pass
        raw = await asyncio.gather(
            *(guarded(session, i) for i in range(total)),
            return_exceptions=True)
        wall = time.perf_counter() - wall_start

    results, failures = [], []
    for r in raw:
        if isinstance(r, BaseException):
            failures.append({"error": type(r).__name__, "detail": str(r)[:200]})
        elif r.get("ok"):
            results.append(r)
        else:
            failures.append(r)
    return results, failures, wall


# ----------------------------------------------------------------------------
# stats
# ----------------------------------------------------------------------------
def pct(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(results, failures, wall, args):
    """Reduce one pass to a flat metric dict."""
    n_ok, n_fail = len(results), len(failures)
    total = n_ok + n_fail

    ttfts = [r["ttft_ms"] for r in results]
    e2es = [r["e2e_ms"] for r in results]
    tpots = [r["tpot_ms"] for r in results]
    itls = [x for r in results for x in r["itls"]]
    out_tokens = sum(r["out_tokens"] for r in results)
    in_tokens = sum(r["in_tokens"] for r in results)

    # Queue delay: with client concurrency above the server's max-num-seqs,
    # the excess TTFT is waiting, not prefill. Reported, not inferred away.
    served = min(args.concurrency, args.server_max_num_seqs or args.concurrency)
    queued = max(args.concurrency - served, 0)

    # Goodput counts a FAILED request as having missed the SLO: a config that
    # drops requests must not score better than one that serves them slowly.
    met = [1 for r in results if r["e2e_ms"] <= args.slo_ms] if args.slo_ms else []
    goodput = (len(met) / total * 100) if (args.slo_ms and total) else float("nan")

    # A pass with no successful request is not a fast pass, it is a failed one.
    # Rates and cost are left undefined so the surrogate model cannot learn a
    # crashed configuration as a cheap one (SLO-Guard, 2026).
    valid = n_ok > 0
    partial = n_ok > 0 and n_fail > 0

    if valid:
        # Rates are based on SERVED requests; cost is amortised over them too.
        throughput = out_tokens / wall if wall else 0.0
        rps = n_ok / wall if wall else 0.0
        cost_per_1k = ((wall / 3600) * args.gpu_cost_per_hour / n_ok) * 1000
    else:
        throughput = rps = 0.0
        cost_per_1k = float("nan")

    return {
        "requests_sent": total,
        "requests_ok": n_ok,
        "requests_failed": n_fail,
        "run_valid": valid,
        "run_partial": partial,
        "wall_s": wall,
        "out_tokens": out_tokens,
        "in_tokens": in_tokens,
        # The mean completion length, which is NOT max_tokens: these prompts
        # stop at EOS around 116 tokens against a 128 limit, and the prefix
        # runs come in between 96 and 111. A cost model that assumes the
        # limit over-predicts every latency by 9-25%, and because the queue
        # term is proportional to the output length it inflates TTFT too.
        # Finding #41. Recorded as a metric so it travels to the ledger.
        "out_tokens_mean": (out_tokens / n_ok) if n_ok else 0.0,
        # The MEDIAN completion length, which is a different number from the
        # mean and is the one a percentile latency needs. Finding #41: using
        # the mean (116) to predict e2e_p50 made it worse, not better,
        # because when most requests run to the max_tokens limit the median
        # request is at the limit while the mean is dragged down by the few
        # that stop early. Cost needs the mean, because cost is total work;
        # a percentile needs the percentile.
        "out_tokens_p50": pct([r["out_tokens"] for r in results], 50),
        "ttft_p50": pct(ttfts, 50), "ttft_p95": pct(ttfts, 95),
        "ttft_p99": pct(ttfts, 99),
        "tpot_p50": pct(tpots, 50), "tpot_p99": pct(tpots, 99),
        "itl_p50": pct(itls, 50), "itl_p99": pct(itls, 99),
        "e2e_p50": pct(e2es, 50), "e2e_p95": pct(e2es, 95),
        "e2e_p99": pct(e2es, 99),
        "e2e_mean": statistics.mean(e2es) if e2es else float("nan"),
        "throughput_tok_s": throughput,
        "rps": rps,
        "goodput_pct": goodput,
        "client_concurrency": args.concurrency,
        "server_max_num_seqs": args.server_max_num_seqs,
        "queued_excess": queued,
        "cost_per_1k": cost_per_1k,
        "token_source": results[0]["token_source"] if results else "none",
    }


def verdict(ok, sent, failed):
    """One word the planner can filter on, before any number is compared."""
    if ok == 0:
        return "FAILED"
    if failed:
        return "PARTIAL"
    return "OK"


def median_of(passes, key):
    vals = [p[key] for p in passes if isinstance(p[key], (int, float))
            and p[key] == p[key]]
    return statistics.median(vals) if vals else float("nan")


def spread_of(passes, key):
    """Relative spread (max-min)/median in percent - the noise estimate."""
    vals = [p[key] for p in passes if isinstance(p[key], (int, float))
            and p[key] == p[key]]
    if len(vals) < 2:
        return 0.0
    med = statistics.median(vals)
    return (max(vals) - min(vals)) / med * 100 if med else 0.0


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="client-side concurrent requests")
    ap.add_argument("--server-max-num-seqs", type=int, default=None,
                    help="the server's --max-num-seqs, for queue accounting")
    ap.add_argument("--requests", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--shared-prefix-tokens", type=int, default=0,
                    help="prepend this many tokens of identical text to every "
                         "prompt; required to measure prefix caching at all")
    ap.add_argument("--repeat", type=int, default=3,
                    help="passes per configuration; median is reported")
    ap.add_argument("--slo-ms", type=float, default=0.0,
                    help="end-to-end SLO for goodput; 0 disables")
    ap.add_argument("--gpu-cost-per-hour", type=float, default=0.35)
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--json-dir", default="runs")
    args = ap.parse_args()

    prefix = shared_prefix(args.shared_prefix_tokens)
    print(f"[{args.label}] {args.repeat} pass(es) x {args.requests} requests "
          f"@ client concurrency {args.concurrency}")
    if prefix:
        print(f"  shared prefix: ~{args.shared_prefix_tokens} tokens "
              f"({len(prefix)} chars) identical on every request")

    passes, all_failures = [], []
    for i in range(args.repeat):
        results, failures, wall = asyncio.run(
            one_pass(args.url, args.model, args.concurrency,
                     args.requests, args.max_tokens, prefix))
        s = summarize(results, failures, wall, args)
        passes.append(s)
        all_failures.extend(failures)
        print(f"  pass {i+1}/{args.repeat}: "
              f"{s['requests_ok']}/{s['requests_sent']} ok, "
              f"{s['throughput_tok_s']:.0f} tok/s, "
              f"e2e p99 {s['e2e_p99']:.0f} ms, "
              f"${s['cost_per_1k']:.4f}/1K")

    keys = ["ttft_p50", "ttft_p95", "ttft_p99", "tpot_p50", "tpot_p99",
            "itl_p50", "itl_p99", "e2e_p50", "e2e_p95", "e2e_p99", "e2e_mean",
            "throughput_tok_s", "rps", "goodput_pct", "cost_per_1k",
            "out_tokens_mean", "out_tokens_p50"]
    med = {k: median_of(passes, k) for k in keys}
    noise = spread_of(passes, "throughput_tok_s")

    ok = sum(p["requests_ok"] for p in passes)
    sent = sum(p["requests_sent"] for p in passes)
    failed = sent - ok
    status = verdict(ok, sent, failed)

    print()
    print("=" * 62)
    print(f"  CONFIG: {args.label}   [{status}]   "
          f"(median of {args.repeat} pass(es))")
    print("=" * 62)
    print(f"  requests ok / sent  : {ok} / {sent}"
          + (f"   FAILED {failed}" if failed else ""))
    if status == "FAILED":
        print("  -> no request succeeded: this configuration is INFEASIBLE.")
        print("     Rates and cost are not reported; the row is kept as a")
        print("     negative observation for the cost model.")
    print(f"  client concurrency  : {args.concurrency}")
    print(f"  server max-num-seqs : {args.server_max_num_seqs or 'unknown'}")
    if args.server_max_num_seqs and args.concurrency > args.server_max_num_seqs:
        print(f"  -> {args.concurrency - args.server_max_num_seqs} request(s) "
              f"queue in excess of server capacity: TTFT includes wait time")
    print(f"  token source        : {passes[0]['token_source']}")
    print("-" * 62)
    print(f"  TTFT  p50/p95/p99   : {med['ttft_p50']:7.1f} /{med['ttft_p95']:8.1f} /{med['ttft_p99']:8.1f} ms")
    print(f"  ITL   p50/p99       : {med['itl_p50']:7.1f} /{med['itl_p99']:8.1f} ms")
    print(f"  TPOT  p50/p99       : {med['tpot_p50']:7.1f} /{med['tpot_p99']:8.1f} ms")
    print(f"  E2E   p50/p95/p99   : {med['e2e_p50']:7.1f} /{med['e2e_p95']:8.1f} /{med['e2e_p99']:8.1f} ms")
    print("-" * 62)
    print(f"  throughput          : {med['throughput_tok_s']:8.1f} tokens/s")
    print(f"  request rate        : {med['rps']:8.2f} req/s")
    if args.slo_ms:
        print(f"  goodput (<= {args.slo_ms:.0f} ms) : {med['goodput_pct']:8.1f} %")
    print(f"  cost per 1K requests: ${med['cost_per_1k']:.4f}")
    print(f"  noise (throughput)  : {noise:8.1f} %  across passes")
    print("=" * 62)

    if all_failures:
        print("  failure breakdown:")
        tally = {}
        for f in all_failures:
            tally[f.get("error", "unknown")] = tally.get(f.get("error", "unknown"), 0) + 1
        for err, n in sorted(tally.items(), key=lambda kv: -kv[1]):
            print(f"    {n:4d}  {err}")

    # ---- CSV row (one per configuration, median values) --------------------
    cols = ["timestamp", "label", "status", "model", "client_concurrency",
            "server_max_num_seqs", "requests_sent", "requests_ok",
            "requests_failed", "max_tokens", "shared_prefix_tokens",
            "repeat"] + keys + ["noise_pct"]
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": args.label, "status": status, "model": args.model,
        "client_concurrency": args.concurrency,
        "server_max_num_seqs": args.server_max_num_seqs or "",
        "requests_sent": sent, "requests_ok": ok, "requests_failed": failed,
        "max_tokens": args.max_tokens,
        "shared_prefix_tokens": args.shared_prefix_tokens,
        "repeat": args.repeat,
        "noise_pct": round(noise, 2),
    }
    row.update({k: round(med[k], 4) if med[k] == med[k] else "" for k in keys})

    # The header is only written for a new file, so adding a column to this
    # list silently leaves an old file's header one field short of every
    # later row. Anything reading that CSV by name or index then gets a
    # different metric than it asked for, with no error.
    #
    # Finding #39: that happened when shared_prefix_tokens was added, and it
    # produced two confidently wrong claims about chunked prefill before the
    # mismatch was noticed. So the header is now checked, and a file whose
    # header does not match is set aside rather than appended to.
    header_line = ",".join(cols)
    existing = None
    if os.path.exists(args.csv):
        try:
            with open(args.csv) as f:
                existing = f.readline().strip()
        except OSError:
            existing = None
    if existing and existing != header_line:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archived = f"{args.csv}.{stamp}.old"
        try:
            os.rename(args.csv, archived)
            print(f"  NOTE: {args.csv} had a header from an older version of")
            print(f"  this script, so its columns no longer match the rows")
            print(f"  being written. It has been moved to {archived} and a")
            print(f"  new file started. The old rows are intact; read them")
            print(f"  with the header that is now at the top of that file.")
            existing = None
        except OSError as e:
            print(f"  WARNING: {args.csv} has a stale header and could not "
                  f"be renamed ({e}).")
            print(f"  Columns in that file do NOT match these rows. Pass "
                  f"--csv with a new path.")
    write_header = existing is None
    with open(args.csv, "a") as f:
        if write_header:
            f.write(header_line + "\n")
        f.write(",".join(str(row[c]) for c in cols) + "\n")
    print(f"  appended to {args.csv}")

    # ---- full JSON record (per-pass detail, for the surrogate model) -------
    os.makedirs(args.json_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(args.json_dir, f"{args.label}_{stamp}.json")
    with open(path, "w") as f:
        json.dump({"args": vars(args), "status": status, "passes": passes,
                   "median": med, "noise_pct": noise,
                   "failures": all_failures[:50]}, f, indent=2)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()