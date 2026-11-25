"""
InferPlan: cost-aware inference planning for LLM serving.

Measures a few serving configurations on the GPU you actually have, learns a
cost model from those measurements, and then prices the rest of the search
space without running it - so you get a `vllm serve` command that meets your
latency and accuracy limits at the lowest cost per request.

What makes it different from a benchmark script is what it refuses to do:

  - It does not recommend a configuration it has never seen measured without
    labelling the figure as an extrapolation.
  - It does not plan at all when the measurements in its ledger do not look
    like measurements.
  - It re-measures a fixed reference configuration during every sweep, and
    throws the sweep away if the machine drifted while it ran. On the laptop
    this was developed on, one configuration read 4730, 3083 and 801 tokens/s
    within 25 hours, each reading repeatable to 0.2%. Without that control,
    a 6x platform effect gets attributed to whichever knob was being turned.

The modules, in the order a session uses them:

    serve        start exactly one vLLM server and wait until it answers
    bench        measure one running server (TTFT, TPOT, ITL, cost, noise)
    runner       sweep configurations, with the reference control
    costmodel    learn from the ledger; leave-one-out validation
    planner      search the space, rank by cost, print the serve command
    quality      measure what quantization costs in output quality
    gpuwatch     sample the GPU during a run and say what it saw
    mergeledger  combine ledgers from several machines or sessions
    repairledger re-pair ledger rows with their own benchmark records
"""

# Read from the installed package metadata, which pyproject.toml is the only
# source of. Hard-coding it here made `inferplan --version` print 0.1.0 while
# the wheel on TestPyPI was 0.1.0a1 - the same version claimed in two places,
# with nothing comparing them. That is the shape of the dependency-list bug
# (finding #38) and the requires-python bug: a fact stated twice is a fact
# that will eventually disagree with itself.
try:
    from importlib.metadata import PackageNotFoundError, version as _version
    try:
        __version__ = _version("inferplan")
    except PackageNotFoundError:
        # Running from a source checkout that was never installed.
        __version__ = "0.0.0+local"
except ImportError:                      # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = [
    "bench", "runner", "costmodel", "planner", "quality",
    "serve", "gpuwatch", "mergeledger", "repairledger",
]
