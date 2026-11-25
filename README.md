# InferPlan

Finds the cheapest way to serve your LLM without breaking your latency limit —
by measuring a few settings on your own GPU and working out the rest.

```bash
pip install inferplan

inferplan sweep --sweep knee --reference-every 4   # measure (about 15 min)
inferplan fit                                      # learn, and show its error
inferplan plan --rate 20 --p99-ms 2000             # decide
```

The last command prints a `vllm serve` line you can paste, how many GPUs that
traffic needs, the cost per 1000 requests, and the command to check it.

---

## Why serving settings matter

vLLM has about a dozen settings that interact: how many requests share the GPU
(`--max-num-seqs`), the number format (`--quantization`, `--dtype`), how much
memory to reserve (`--gpu-memory-utilization`), whether to cache repeated
prompts, and more.

Pick them badly and you pay two or three times more than you need to, or you
blow your latency budget. Testing them honestly means dozens of benchmark runs,
and the right answer changes with your model, your prompt shape, and your card.
So most teams try three or four combinations, pick one that works, and move on —
which is a reasonable thing to do and usually leaves money on the table.

InferPlan measures a handful of settings, learns how your GPU behaves from
those, and then prices the combinations it did **not** run.

## What it actually does

**1. Measures.** It starts a vLLM server, sends it real streaming requests, and
records time-to-first-token, time-per-output-token, end-to-end latency, how many
requests it can handle per second, and the cost. Then it stops the server and
does the next setting.

**2. Checks that your machine stayed still.** Before, during and after every
sweep it re-measures one fixed setting. If that reference drifts, the sweep is
thrown away. This matters more than it sounds — see below.

**3. Learns three things** from the measurements, each fitted where its own
effect dominates:

- **How long one decode step takes**, as the number of requests sharing it grows:
  `step = c0 + c1 × batch^alpha`. `c0` turns out to be the time to read the
  model weights once, which is what physics says it should be.
- **How long the first token takes**, split between prompt tokens the cache had
  to compute and prompt tokens it served from memory.
- **How long requests wait** when more arrive than the server has room for.
  Queueing theory predicts a coefficient of 1.0; measured, it came out 0.99.

**4. Decides.** You give it your constraints — traffic rate, latency limit,
accuracy you can afford to lose, budget — and it searches the whole space,
ranks by cost, and prints the command. It counts GPUs in whole units, because
a GPU is rented by the hour, not by the request.

## Installing and running it

**Install:**

```bash
pip install inferplan              # the tools
pip install inferplan[serve]       # and vLLM, if it should start servers too
```

You need a machine with an NVIDIA GPU, `nvidia-smi`, and Python 3.10 or newer.
On Windows this means WSL2, because vLLM needs Linux.

**Step 1 — measure (about 15 minutes).** Nine settings, each benchmarked three
times, with a reference check every fourth one:

```bash
inferplan sweep --sweep knee --reference-every 4
```

Leave the machine otherwise idle while this runs. At the end it tells you
whether the reference held steady; if it did not, run it again after a rest
rather than using the numbers.

**Step 2 — learn.** No GPU needed; this reads the measurements from step 1:

```bash
inferplan fit
```

This prints what it learned and how wrong it is, holding out each run in turn.
Read the error table before trusting anything below it.

**Step 3 — decide.** Describe your traffic and your limits:

```bash
inferplan plan --rate 20 --input-tokens 512 --output-tokens 128 --p99-ms 2000
```

- `--rate` requests per second you expect
- `--input-tokens` / `--output-tokens` your typical prompt and reply length
- `--p99-ms` the latency ceiling you must not cross
- `--prefix-overlap 0.8` if most of your prompt repeats (a system prompt, say)
- `--budget-per-1k`, `--max-gpus`, `--max-accuracy-drop-pct` if those bind

It prints the `vllm serve` command, the GPU count, the cost, and a verify
command.

**Step 4 — verify before you deploy.** The plan ends with the exact command;
run it. A prediction is an estimate until measured:

```bash
inferplan sweep --slots 32 --reference-every 1 --repeat 3
```

**Measuring more** makes the predictions better. Each of these adds a dimension:

```bash
inferplan sweep --sweep slots --reference-every 1      # bigger batches
inferplan sweep --sweep prefix --reference-every 4     # prompt caching
inferplan sweep --sweep prefix_curve --reference-every 4
inferplan sweep --sweep quant --reference-every 4      # number formats
inferplan sweep --sweep kvcache --reference-every 4
```

**Quality,** if you are considering a quantized model. Start a server, then:

```bash
inferplan quality --label fp16_baseline --set-baseline
# restart with the quantized build, then:
inferplan quality --label int4
```

**If something goes wrong:**

```bash
inferplan serve --stop        # a server was left running
inferplan watch --seconds 60  # the GPU is behaving oddly
inferplan repair              # the measurements look mismatched
inferplan <command> --help    # every command explains itself
```

## Why the "reference control" is the interesting part

While building this, one single setting measured **4730, 3083 and 801 tokens/s
within 25 hours** on the same laptop — and each of those readings repeated to
within 0.2% across three passes. Nothing in the configuration changed. The
machine itself drifted, and recovered fully after fifteen minutes idle.

Someone testing one setting per sitting over an afternoon would have credited a
6x platform effect to whichever knob they happened to be turning, with excellent
internal consistency to back it up. That is not carelessness; it is unavoidable
without a control that returns to a previous setting and checks.

So every sweep brackets its measurements with a reference reading. If the
machine moved, you are told while the sweep is still running, not a day later.
Automating the sweep is convenience. **Automating the control is the point.**

## How to check that it isn't lying to you

Fair question to ask of anything that predicts a number. Four ways, all built in:

**It scores itself.** `inferplan fit` holds out each measured run, refits
without it, predicts it, and reports the error — so the score is on runs the
model never saw. From a development machine (RTX 3050, Qwen2.5-0.5B):

```
TPOT                    1.9 %
                 median miss 0.15 ms on a typical 8.3 ms
TTFT                   17.3 %       OVER TARGET
                 median miss 4.60 ms on a typical 40.0 ms
E2E latency             2.3 %
                 median miss 22.90 ms on a typical 1058.1 ms
cost per 1K             8.1 %
                 median miss $0.00065 on a typical $0.0112
```

Your numbers will differ, because they describe your GPU and your model — which
is the whole idea. Run `inferplan fit` and read your own.

The absolute miss is printed next to every percentage on purpose. TTFT's 17%
looks alarming until you see it is 4.6 ms on a 40 ms quantity — about one decode
step, and below any plausible scheduling jitter. A percentage without a
magnitude is a verdict without evidence.

**It tells you its own noise floor.** If repeated runs of the same setting
disagree by 5%, a 3% difference between two settings is not a result, and it
says so rather than ranking them.

**It refuses when it should.** It will not quietly recommend something it has
never measured: configurations outside the measured range are labelled as
extrapolations and carry a penalty, configurations whose coefficients were
borrowed from a neighbour rank *below* measured ones even when they price
cheaper, and if the ledger contains runs that do not look like measurements, the
planner exits instead of producing a confident wrong answer.

**It hands you the check.** Every recommendation ends with the exact command to
verify it by measurement, run under the reference control.

## Commands

| command | what it does |
|---|---|
| `inferplan serve MODEL` | start exactly one server and wait until it answers |
| `inferplan bench` | measure the running server |
| `inferplan sweep` | measure many settings, with the reference control |
| `inferplan fit` | learn from the measurements and report the error |
| `inferplan plan` | search and recommend, with provenance |
| `inferplan quality` | measure what a number format costs in output quality |
| `inferplan watch` | sample the GPU during a run and say what it saw |
| `inferplan merge` | combine measurements from several machines |
| `inferplan repair` | re-pair measurements with their own records |

Each takes `--help`.

## License

MIT.
