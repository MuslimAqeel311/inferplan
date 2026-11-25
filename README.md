# InferPlan

Picks the cheapest vLLM serving configuration that still meets your latency
and accuracy limits — by measuring a handful on the GPU you actually have and
pricing the rest without running them.

```
pip install inferplan

inferplan sweep --sweep knee --reference-every 4     # measure
inferplan fit                                        # learn, and validate
inferplan plan --rate 20 --p99-ms 2000               # decide
```

The last command prints a `vllm serve` line, how many GPUs that rate needs,
the cost per 1000 requests, and the benchmark command to verify it.

## Why not just benchmark it yourself

You can, and you should check whatever this recommends. The problem is not
running a benchmark; it is that a serving configuration has around a dozen
interacting knobs, so trying them honestly means dozens of runs, and the
answer changes with your model, your prompt shape and your card.

There is also a failure that is easy to miss by hand. During development, one
single configuration measured **4730, 3083 and 801 tokens/s within 25 hours**
on the same laptop — each of those readings repeatable to 0.2% across three
passes. Nothing in the configuration changed; the platform drifted. Someone
testing one setting per sitting over an afternoon would have attributed a 6x
platform effect to whichever knob they happened to be turning, with excellent
internal consistency to back it up.

So every sweep re-measures a fixed **reference configuration** at the start,
periodically, and at the end. If the reference moves more than your tolerance,
the sweep rests and re-checks; if it still moves, the sweep is marked
contaminated and stops. Automating the sweep is convenience. Automating the
control is the point.

## What it refuses to do

- **Recommend an unmeasured configuration silently.** Anything beyond the
  measured range is labelled an extrapolation and carries a haircut.
- **Plan on data that is not measurements.** If the ledger has runs whose own
  passes disagree, or inverted results, `plan` exits rather than producing a
  confident wrong answer.
- **Treat an uncontrolled run as evidence.** Benchmarks run by hand are
  recorded and can raise a caution, but they never cap or set a ceiling.
- **Hide its own error.** `inferplan fit` reports leave-one-out MAPE — the
  model scored on runs it was not fitted to — next to the session's noise
  floor, so you can tell model error from measurement error.
- **Claim a quality result it cannot resolve.** `inferplan quality` prints the
  smallest difference its corpus could have detected alongside every number.

## Commands

| command | what it does |
|---|---|
| `inferplan serve MODEL` | start exactly one server and wait until it answers |
| `inferplan bench` | measure the running server: TTFT, TPOT, ITL, goodput, cost, noise |
| `inferplan sweep` | measure many configurations, with the reference control |
| `inferplan fit` | learn the cost model from the ledger; leave-one-out validation |
| `inferplan plan` | search the space and recommend, with provenance |
| `inferplan quality` | measure what a precision costs in output quality |
| `inferplan watch` | sample the GPU during a run and say what it saw |
| `inferplan merge` | combine ledgers from several machines or sessions |

Each takes `--help`.

## What the model is

Three learned pieces, each fitted where its own effect dominates:

- **Decode step** — `step_ms = c0 + c1 · batch^alpha`, where the batch is
  `min(concurrency, max_num_seqs)`, not the slot limit. `c0` lands on the
  roofline step time (weights read once per step, whatever the batch) and the
  rest is per-sequence work growing faster than the batch.
- **Prefill** — `TTFT = c0 + c1 · full_price_tokens^alpha + c2 · cached`,
  which separates the prompt the cache had to compute from the prompt it
  served.
- **Queueing** — Little's Law with one fitted coefficient; it came out at
  0.99 against the 1.0 the law predicts.

Capacity is then counted in whole GPUs, because a GPU is rented by time, not
by request.

## Honest limits

Everything above was measured on **one** laptop GPU (RTX 3050, 4 GB) with
**one** model (Qwen2.5-0.5B) over batches of 4 to 128. On that machine the
planner predicts throughput to about 5%, latency to about 7% and capacity to
about 6%, with every input's provenance attached. None of that establishes
the same accuracy on another card, another model or another prompt shape —
those are the next measurements, not inferences, and the library is built to
tell you when it is extrapolating rather than to pretend it is not.

Known gaps, stated rather than hidden:

- Chunked prefill inflates inter-token latency on long prompts and the decode
  model does not include that term yet, so TTFT on long-prompt workloads is
  the weakest prediction in the set.
- Isolating what a quantization costs needs the same checkpoint served at two
  precisions, which does not fit in 4 GB; the current quality numbers compare
  two different checkpoints and say so.

## Requirements

Python 3.9+, `nvidia-smi`, and a vLLM server — either one you start yourself
or one `inferplan serve` starts for you (`pip install inferplan[serve]`).
No numpy; the least squares is about forty lines.

## License

MIT.
