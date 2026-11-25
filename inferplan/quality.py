#!/usr/bin/env python3
"""
InferPlan quality: measure what compression costs in output quality.

The planner has been reporting accuracy drops from a table of plausible
numbers, labelled as placeholders. This module replaces them with
measurements.

Two things make this cheap, and both come from how quantization works rather
than from how the GPU works:

  1. Quality depends on how far the WEIGHTS were compressed, not on which
     chip runs them. The same INT4 checkpoint gives the same answers on a
     laptop GPU and on an A100, so a measurement taken here transfers.
  2. Quality does not depend on batch size, caching or concurrency. Those
     change when a token is computed, not which token is chosen. So a search
     space of hundreds of configurations needs only one quality run per
     (model, quantization) pair - six, not six hundred.

WHAT IS MEASURED
  Perplexity over a fixed passage, and accuracy on a small multiple-choice
  set, both scored from the token log-probabilities the server returns.

  The absolute numbers are not comparable to published benchmarks: the corpus
  is small and embedded in this file. What IS valid is the RELATIVE change
  against an fp16 baseline measured on the same text, and that is the number
  the planner needs. The report says so rather than implying a benchmark.

Usage
    # with a server already running
    python quality.py --label fp16_baseline --set-baseline

    # then against a quantized server
    python quality.py --label awq_int4
"""
import argparse
import json
import math
import statistics
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_URL = "http://localhost:8000"

# ----------------------------------------------------------------------------
# the corpus, embedded so no download is needed and the text never changes
# ----------------------------------------------------------------------------
PASSAGES = [
    "Memory bandwidth, not arithmetic throughput, sets the pace of "
    "autoregressive decoding. Each step reads every weight once to produce a "
    "single token, so a model that fits in fast memory decodes quickly while "
    "one that spills into slower memory does not, regardless of how many "
    "arithmetic units the processor has.",

    "A key-value cache stores the attention keys and values computed for "
    "tokens already processed, so that each new token attends to the past "
    "without recomputing it. The cache grows with sequence length, and on a "
    "small accelerator it becomes the binding constraint long before the "
    "weights do.",

    "Continuous batching admits a new request as soon as a slot frees, rather "
    "than waiting for a whole batch to finish. The effect is that throughput "
    "rises with the number of slots while the latency of any single request "
    "stays close to what it would be alone, until the slots themselves run "
    "out.",

    "Quantization reduces the number of bits used to store each weight. "
    "Fewer bits mean less memory traffic per decoding step and therefore "
    "faster generation, at the cost of some precision in the represented "
    "values. Whether that cost is visible in the output depends on the task.",

    "The tail of a latency distribution matters more than its centre for "
    "services that people wait on. A median of two hundred milliseconds with "
    "a ninety-ninth percentile of four seconds feels slow, because the slow "
    "requests are the ones a user notices and remembers.",

    # The passages below were added to bring the corpus to roughly fifteen
    # hundred tokens. At the original two hundred and sixty-six, top-1
    # accuracy carried an interval of plus or minus five points, wide enough
    # to miss a genuine three point change. The arithmetic is in finding 27.
    "Prefill and decode make different demands on a processor. Prefill "
    "processes every prompt token at once and so has plenty of parallel work, "
    "which keeps the arithmetic units busy. Decode produces one token at a "
    "time and spends most of its effort moving weights, which leaves those "
    "same units idle while memory catches up.",

    "Throughput and latency pull against each other in a serving system. "
    "Larger batches amortise the cost of reading weights across more "
    "requests, which raises the number of tokens produced each second. The "
    "same batching makes any individual request wait for its companions, so "
    "the time a single user waits goes up as the server gets more efficient.",

    "A roofline model draws two limits on one chart: the rate at which a "
    "processor can move data, and the rate at which it can compute. Any "
    "workload sits under both lines. Which line it touches first tells you "
    "whether buying faster memory or more arithmetic units would help, and "
    "usually only one of the two would.",

    "Measuring a system once tells you very little. Repeating the same "
    "measurement shows how much the number moves when nothing has changed, "
    "and that spread is what any difference must be compared against. A gap "
    "smaller than the spread is not a result; it is the instrument reporting "
    "its own unsteadiness.",

    "Consumer graphics cards reserve part of their memory for the display. "
    "How much depends on what windows are open, so the amount left for a "
    "model is not fixed and can move by hundreds of megabytes between one "
    "session and the next. A configuration that starts today may refuse to "
    "start tomorrow with nothing changed.",

    "Power limits shape what a laptop processor can sustain. A chip may reach "
    "its rated clock speed briefly and then settle lower once the power "
    "budget is reached, so a short test flatters the hardware. Any figure "
    "taken from a long run already includes that settling, which is usually "
    "what a deployment would see.",

    "Quantization schemes differ in where they lose precision. Some round "
    "every weight the same way, others keep a scale for each small group of "
    "weights so that outliers do not drag the rest down. The second costs a "
    "little memory for the scales and usually preserves more of the original "
    "behaviour.",

    "A serving engine chooses among several attention implementations at "
    "startup, based on what the hardware supports. Two servers given "
    "identical flags can therefore run different code, and a measurement that "
    "does not record which implementation was selected cannot explain why two "
    "supposedly identical runs disagreed.",

    "Compiling a model graph ahead of time removes work from every later "
    "request, but the compilation itself takes tens of seconds. Caching the "
    "result makes the first start slow and the rest quick, which matters when "
    "a study restarts a server hundreds of times and would otherwise spend "
    "most of its hours compiling.",

    "Little's Law relates three quantities in any queue: the number of items "
    "in the system, the rate at which they arrive, and how long each one "
    "stays. Knowing two of them fixes the third. It needs no assumption about "
    "the distribution of arrivals, which makes it unusually safe to apply to "
    "a serving system.",

    "An orphaned process can hold memory on an accelerator long after the "
    "program that started it appears to have stopped. The next program then "
    "starts with less memory than it expected and performs badly for reasons "
    "that have nothing to do with its own settings, which is a difficult "
    "fault to diagnose from its numbers alone.",

    "Extrapolating a trend beyond the range that was measured is a claim, not "
    "a measurement. A curve fitted to three points can be continued in "
    "several directions that all agree where the data lies and disagree "
    "sharply beyond it, so the honest treatment is to mark such a figure and "
    "then go and measure it.",

    "Perplexity summarises how surprised a model was by a passage of text. "
    "Lower values mean the text was more predictable to the model. Because it "
    "reads the whole probability distribution rather than only the top "
    "choice, it responds to changes in confidence that a simple accuracy "
    "count would not notice.",

    "A confidence interval states what a measurement could and could not have "
    "seen. A result reported without one invites the reader to treat a wide "
    "guess as a firm finding, and a null result without one is especially "
    "misleading, since it may be reporting the resolution of the instrument "
    "rather than the behaviour of the system.",

    "Caching the opening section of a prompt helps only when that section "
    "repeats. Traffic that sends a long standing instruction with every "
    "request benefits greatly; traffic made of unrelated short questions has "
    "nothing to reuse and gains almost nothing, which is why the same setting "
    "can be essential or pointless.",

    "Scheduling decides which requests occupy the processor at any moment. A "
    "scheduler that admits work as soon as room appears keeps the hardware "
    "busier than one that waits to assemble a full group, and the difference "
    "shows up as shorter waits for the requests that would otherwise have sat "
    "in the queue.",

    "Memory for attention state is allocated in pages rather than reserved in "
    "full for each request. A sequence therefore occupies only the pages it "
    "has actually filled, which allows many more concurrent requests than a "
    "scheme that set aside the maximum possible length for every one of them.",

    "Reporting a cost per request on a single processor can mislead, because "
    "the processor is rented by time rather than by request. What a "
    "configuration really decides is how many processors a given arrival rate "
    "needs, and that is counted in whole machines rather than in fractions of "
    "a cent.",

    "A benchmark that reports only averages hides the behaviour that users "
    "complain about. Percentiles show the shape of the distribution, and the "
    "upper ones describe the experience of the unlucky minority whose "
    "requests arrived when the system was busiest and who are most likely to "
    "remember it.",

    "Holding a configuration fixed while changing one knob is the only way to "
    "attribute a change to that knob. Studies that move several at once "
    "produce a number that cannot be explained afterwards, and the effort "
    "spent collecting it buys no understanding of which setting mattered.",

    "Hardware differs in ways that a single scaling factor cannot capture. "
    "One card may have twice the memory bandwidth and the same arithmetic "
    "throughput as another, so a workload limited by memory improves while a "
    "workload limited by computation does not move at all.",

    "Two measurements that disagree are more informative than two that agree. "
    "The disagreement points at an assumption that one of them relies on, and "
    "finding which assumption is wrong usually teaches more about the system "
    "than either number would have done on its own.",
]

# Completion questions scored by log-probability, not by generated text, so
# the measurement does not depend on the model following an output format.
CHOICES = [
    {"context": "During autoregressive decoding, the time for one step is "
                "usually limited by",
     "choices": ["reading the model weights from memory",
                 "the number of arithmetic units on the chip",
                 "the speed of the network interface",
                 "the size of the operating system page cache"],
     "answer": 0},
    {"context": "A key-value cache grows as a sequence gets longer because it "
                "stores",
     "choices": ["attention keys and values for every past token",
                 "a copy of the model weights for each request",
                 "the gradients needed for the backward pass",
                 "a compressed index of the training corpus"],
     "answer": 0},
    {"context": "Raising the number of concurrent sequences a server will "
                "serve tends to",
     "choices": ["increase throughput and increase tail latency",
                 "decrease throughput and decrease tail latency",
                 "leave both throughput and latency unchanged",
                 "increase throughput and decrease tail latency"],
     "answer": 0},
    {"context": "Quantizing a model to fewer bits per weight mainly speeds up "
                "decoding because it",
     "choices": ["reduces the memory traffic of each decoding step",
                 "allows more arithmetic units to be used at once",
                 "shortens the prompt that must be processed",
                 "removes layers from the network"],
     "answer": 0},
    {"context": "Prefix caching helps most when requests",
     "choices": ["share a long identical opening section",
                 "are all completely different from each other",
                 "produce very long outputs",
                 "arrive one at a time with gaps between them"],
     "answer": 0},
    {"context": "The ninety-ninth percentile of a latency distribution "
                "describes",
     "choices": ["the slowest one percent of requests",
                 "the average request",
                 "the fastest one percent of requests",
                 "the total number of requests served"],
     "answer": 0},
    {"context": "A server that reports a low internal variance across repeated "
                "runs has shown that its measurements are",
     "choices": ["repeatable, which does not make them correct",
                 "correct, because they agree with each other",
                 "faster than a server with higher variance",
                 "independent of the hardware they ran on"],
     "answer": 0},
    {"context": "If throughput stops rising as more batch slots are added, the "
                "system has most likely reached",
     "choices": ["a ceiling set by power or compute",
                 "the end of its training data",
                 "the maximum number of network connections",
                 "the limit of its vocabulary size"],
     "answer": 0},
]


# ----------------------------------------------------------------------------
# scoring through the server's log-probabilities
# ----------------------------------------------------------------------------
def post(url: str, path: str, payload: dict, timeout: float = 300) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{url}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def raw_token_logprobs(url: str, model: str, text: str) -> list:
    """
    Per-token log-probabilities for `text`, POSITIONS INTACT.

    Asking for zero new tokens with echo on makes the server return the
    prompt's own log-probabilities, which is what scoring needs: no sampling
    is involved, so the measurement does not depend on temperature or on the
    model following any output format.

    The first entry is None, because the first token has nothing to condition
    on. It is kept here so that index i still corresponds to token i; dropping
    it would shift every later index by one and quietly misalign any slice.
    """
    out = post(url, "/v1/completions", {
        "model": model, "prompt": text,
        "max_tokens": 0, "echo": True, "logprobs": 0, "temperature": 0.0,
    })
    lp = out["choices"][0].get("logprobs") or {}
    return list(lp.get("token_logprobs") or [])


def token_logprobs(url: str, model: str, text: str) -> list:
    """Scorable log-probabilities, with the unconditioned first token dropped."""
    return [v for v in raw_token_logprobs(url, model, text) if v is not None]


def count_tokens(url: str, model: str, text: str) -> int:
    """How many tokens `text` is, from the server's own tokenizer."""
    out = post(url, "/tokenize", {"model": model, "prompt": text})
    if "count" in out:
        return int(out["count"])
    return len(out.get("tokens") or [])


def scored_passage(url: str, model: str, text: str, top_k: int = 5) -> tuple:
    """
    (token logprobs, per-position top-1 hit flags) for one passage.

    One request produces both numbers, because the server can return the
    alternatives it considered at each position alongside the token that was
    actually there. A position counts as a hit when the real next token was
    the model's own first choice.
    """
    out = post(url, "/v1/completions", {
        "model": model, "prompt": text, "max_tokens": 0, "echo": True,
        "logprobs": top_k, "temperature": 0.0,
    })
    lp = out["choices"][0].get("logprobs") or {}
    vals = lp.get("token_logprobs") or []
    tops = lp.get("top_logprobs") or []

    logprobs, hits = [], []
    for i, v in enumerate(vals):
        if v is None:                       # first token has no context
            continue
        logprobs.append(v)
        alts = tops[i] if i < len(tops) else None
        if isinstance(alts, dict) and alts:
            # A hit means nothing scored better than what was actually there.
            hits.append(v >= max(alts.values()) - 1e-9)
    return logprobs, hits


def perplexity_and_top1(url: str, model: str, passages: list) -> tuple:
    """
    (perplexity, tokens, top1 hits, top1 trials).

    Both come out of the same requests. Perplexity says how surprised the
    model was on average; top-1 accuracy says how often its first choice was
    right. Quantization damages the probability ranking, and the second looks
    straight at that.

    This replaced an eight-question multiple-choice set whose 95% interval
    spanned 2% to 47% and therefore could not distinguish any two precisions.
    The same corpus yields a few hundred token-level trials instead, which
    narrows the interval to a few points without a single extra request.
    """
    total, n, hits, trials = 0.0, 0, 0, 0
    for p in passages:
        lps, hs = scored_passage(url, model, p)
        total += sum(lps)
        n += len(lps)
        hits += sum(hs)
        trials += len(hs)
    if not n:
        raise RuntimeError("the server returned no token log-probabilities; "
                           "it may not support echo with logprobs")
    return math.exp(-total / n), n, hits, trials


def perplexity(url: str, model: str, passages: list) -> tuple:
    """(perplexity, token count). Kept for callers that want only this."""
    ppl, n, _, _ = perplexity_and_top1(url, model, passages)
    return ppl, n


def choice_accuracy(url: str, model: str, items: list) -> tuple:
    """
    (accuracy, per-item detail), scoring ONLY the continuation.

    Each choice is scored by the mean log-probability of ITS OWN tokens, given
    the context. The context's tokens are excluded.

    A first version averaged over the whole "context + choice" string and
    scored BELOW chance on a real model: 12.5% where four choices give 25%.
    The reason is arithmetic, not the model. The context is identical across
    the four choices and roughly twice their length, so two thirds of that
    mean was the same text every time and the distinguishing third was
    diluted. Because the choices differ in length, the shared context was
    also weighted differently for each, which left the ranking driven by how
    long a choice was rather than by whether it was right.

    Excluding the context is what lm-evaluation-harness and similar tools do,
    for the same reason.
    """
    correct, detail = 0, []
    for item in items:
        n_ctx = count_tokens(url, model, item["context"])
        scores = []
        for choice in item["choices"]:
            text = f"{item['context']} {choice}"
            raw = raw_token_logprobs(url, model, text)
            # Positions are intact, so everything from n_ctx onward belongs to
            # the choice. Drop the leading None if the slice happens to catch
            # it (it only can when the context is empty).
            cont = [v for v in raw[n_ctx:] if v is not None]
            if not cont:                      # tokenizer disagreement; fall
                cont = [v for v in raw if v is not None][-1:]   # back to last
            scores.append(sum(cont) / len(cont))
        picked = max(range(len(scores)), key=lambda i: scores[i])
        ok = picked == item["answer"]
        correct += ok
        # The margin says how close the call was; a model can be right by a
        # hair and that is worth seeing when comparing precisions.
        ordered = sorted(scores, reverse=True)
        detail.append({"context": item["context"][:48], "picked": picked,
                       "correct": ok,
                       "margin": ordered[0] - ordered[1] if len(ordered) > 1
                       else 0.0})
    return correct / len(items), detail


# ----------------------------------------------------------------------------
# the store the planner reads
# ----------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """
    Wilson 95% interval for a proportion.

    Reported because a small question set produces intervals wide enough to be
    useless, and that has to be visible rather than implied. On the first real
    run, 1 correct out of 8 looked alarming; in fact a model sitting exactly
    at chance scores 1 or fewer 37% of the time, so the number carried no
    information at all. An accuracy printed without its interval invites
    exactly that misreading.
    """
    if n == 0:
        return 0.0, 1.0
    ph = k / n
    d = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / d
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def load_store(path: Path) -> dict:
    if not path.exists():
        return {"baseline": None, "measurements": {}}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"baseline": None, "measurements": {}}


def save_store(path: Path, store: dict):
    path.write_text(json.dumps(store, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=None,
                    help="model id; taken from the server if omitted")
    ap.add_argument("--label", required=True,
                    help="name for this measurement, e.g. awq_int4")
    ap.add_argument("--quantization", default=None,
                    help="the quantization this label represents; the planner "
                         "keys its accuracy table on this")
    ap.add_argument("--set-baseline", action="store_true",
                    help="record this run as the reference to compare against")
    ap.add_argument("--store", default="quality.json")
    ap.add_argument("--skip-choices", action="store_true")
    ap.add_argument("--skip-perplexity", action="store_true")
    args = ap.parse_args()

    model = args.model
    if not model:
        try:
            with urllib.request.urlopen(f"{args.url}/v1/models", timeout=10) as r:
                model = (json.loads(r.read()).get("data") or [{}])[0].get("id")
        except (urllib.error.URLError, OSError, ValueError, IndexError):
            pass
    if not model:
        print(f"no server answering at {args.url}; start one first")
        return 1

    print(f"scoring {model} at {args.url}")
    print(f"  label        : {args.label}")
    print(f"  quantization : {args.quantization or 'none given'}")
    print()

    result = {"label": args.label, "model": model,
              "quantization": args.quantization,
              "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    if not args.skip_perplexity:
        try:
            ppl, ntok, hits, trials = perplexity_and_top1(
                args.url, model, PASSAGES)
        except Exception as e:
            print(f"  perplexity failed: {type(e).__name__}: {e}")
            print("  vLLM needs echo + logprobs on /v1/completions for this.")
            return 2
        result["perplexity"] = ppl
        result["perplexity_tokens"] = ntok
        print(f"  perplexity   : {ppl:.4f}  over {ntok} tokens")
        if trials:
            acc = hits / trials
            lo, hi = wilson(hits, trials)
            result["top1_accuracy"] = acc
            result["top1_trials"] = trials
            result["top1_ci"] = [lo, hi]
            print(f"  top-1 accuracy: {acc:.1%}  ({hits}/{trials} tokens, "
                  f"95% {lo:.0%} to {hi:.0%})")
            # Finding #27, action item 2: never quote a top-1 number without
            # saying what it could not have seen. Two independent samples of
            # this size resolve a difference of about this many points; a
            # smaller gap between two configurations is not a result.
            mde = 1.96 * math.sqrt(2 * acc * (1 - acc) / trials) * 100
            result["top1_min_detectable_pp"] = mde
            print(f"  resolves     : differences of {mde:.1f}pp or larger "
                  f"between two runs of this size;")
            print(f"                 anything smaller this corpus cannot see.")
        else:
            print(f"  top-1 accuracy: unavailable "
                  f"(the server returned no alternatives per position)")

    if not args.skip_choices:
        acc, detail = choice_accuracy(args.url, model, CHOICES)
        n_correct = sum(d["correct"] for d in detail)
        chance = statistics.mean(1 / len(i["choices"]) for i in CHOICES)
        result["choice_accuracy"] = acc
        result["choice_n"] = len(CHOICES)
        result["chance_level"] = chance
        result["mean_margin"] = statistics.mean(d["margin"] for d in detail)
        print(f"  accuracy     : {acc:.1%}  ({n_correct}/{len(CHOICES)} "
              f"correct, chance is {chance:.1%})")
        print(f"  mean margin  : {result['mean_margin']:.4f} "
              f"(how decisively the right choice won)")
        lo, hi = wilson(n_correct, len(CHOICES))
        result["accuracy_ci"] = [lo, hi]
        print(f"  95% interval : {lo:.0%} to {hi:.0%}")
        if lo <= chance <= hi:
            print()
            print(f"  This interval contains chance ({chance:.0%}), so the "
                  f"accuracy figure")
            print(f"  says nothing on its own. With {len(CHOICES)} questions "
                  f"it cannot: even")
            print(f"  a perfect score would only narrow it to "
                  f"{wilson(len(CHOICES), len(CHOICES))[0]:.0%}-100%.")
            print(f"  Use the perplexity comparison instead; it is measured "
                  f"over")
            print(f"  hundreds of tokens rather than {len(CHOICES)} answers.")
            result["accuracy_underpowered"] = True

    store_path = Path(args.store)
    store = load_store(store_path)
    store["measurements"][args.label] = result
    if args.set_baseline:
        store["baseline"] = args.label
        print()
        print(f"  recorded as the baseline")
    save_store(store_path, store)

    base_label = store.get("baseline")
    base = store["measurements"].get(base_label) if base_label else None

    print()
    print("=" * 70)
    if base and base_label != args.label:
        print(f"  AGAINST THE BASELINE  ({base_label})")
        print("=" * 70)
        if base.get("model") and base["model"] != result["model"]:
            print(f"  DIFFERENT MODELS, so this is not a quantization result.")
            print(f"    baseline : {base['model']}")
            print(f"    this run : {result['model']}")
            print(f"  The difference below mixes model size with precision.")
            print(f"  To isolate what the quantization costs, both runs must")
            print(f"  use the same checkpoint at different precisions.")
            print("-" * 70)
            result["cross_model_comparison"] = True
            result.pop("accuracy_drop_pct", None)
        if "perplexity" in result and "perplexity" in base:
            d = (result["perplexity"] / base["perplexity"] - 1) * 100
            print(f"  perplexity : {base['perplexity']:.4f} -> "
                  f"{result['perplexity']:.4f}   {d:+.2f}%")
            result["perplexity_increase_pct"] = d
        if "top1_accuracy" in result and "top1_accuracy" in base:
            d = (base["top1_accuracy"] - result["top1_accuracy"]) * 100
            bl, bh = base.get("top1_ci", (0, 1))
            rl, rh = result.get("top1_ci", (0, 1))
            print(f"  top-1 acc  : {base['top1_accuracy']:.1%} -> "
                  f"{result['top1_accuracy']:.1%}   {-d:+.1f} percentage points")
            overlap = not (rh < bl or bh < rl)
            if overlap:
                print(f"               the two intervals overlap "
                      f"({bl:.0%}-{bh:.0%} vs {rl:.0%}-{rh:.0%}),")
                print(f"               so this is not yet a demonstrated change")
            else:
                print(f"               intervals do not overlap "
                      f"({bl:.0%}-{bh:.0%} vs {rl:.0%}-{rh:.0%}): a real change")
                result["accuracy_drop_pct"] = d
            store["measurements"][args.label] = result
            save_store(store_path, store)

        if "choice_accuracy" in result and "choice_accuracy" in base:
            d = (base["choice_accuracy"] - result["choice_accuracy"]) * 100
            print(f"  accuracy   : {base['choice_accuracy']:.1%} -> "
                  f"{result['choice_accuracy']:.1%}   "
                  f"{-d:+.1f} percentage points")
            if result.get("accuracy_underpowered") or base.get("accuracy_underpowered"):
                print(f"               this difference is inside the "
                      f"uncertainty of both scores;")
                print(f"               it is not evidence of a quality change, "
                      f"and no drop is recorded")
                # Deliberately NOT stored. Writing 0.0 here would let the
                # planner print "accuracy drop 0.0% (measured)", claiming a
                # measurement that was never made. Not measuring something and
                # measuring it to be zero are different states, and only one
                # of them should look confident.
                result.pop("accuracy_drop_pct", None)
            else:
                result["accuracy_drop_pct"] = d
            store["measurements"][args.label] = result
            save_store(store_path, store)
        print("-" * 70)
        print("  Perplexity and top-1 accuracy both rest on a few hundred")
        print("  token-level trials, so both carry real weight. The multiple")
        print("  choice score, if shown, does not: eight questions cannot")
        print("  separate two precisions.")
        print()
        print("  These are RELATIVE figures on a small embedded corpus. They")
        print("  are valid for comparing precisions of the same model, which")
        print("  is what the planner needs. They are not comparable to")
        print("  published benchmark scores and should not be quoted as such.")
        n = result.get("choice_n", 0)
        if n and n < 50:
            print(f"  With {n} questions, one flipped answer moves accuracy by")
            print(f"  {100 / n:.0f} percentage points, so treat small gaps as noise.")
    elif base_label == args.label:
        print(f"  BASELINE RECORDED  ({args.label})")
        print("=" * 70)
        print("  Now start a server with a different precision and run this")
        print("  again with a new --label to get the comparison.")
    else:
        print("  NO BASELINE YET")
        print("=" * 70)
        print("  Run this against an fp16 server with --set-baseline first;")
        print("  the planner needs a reference to measure drops against.")
    print("=" * 70)
    print(f"  stored in {store_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())