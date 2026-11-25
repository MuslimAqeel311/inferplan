#!/usr/bin/env python3
"""
One entry point for the whole library: `inferplan <command> [options]`.

Each command forwards its arguments untouched to the module that owns it, so
`inferplan plan --rate 20` and `python -m inferplan.planner --rate 20` behave
identically. The dispatcher deliberately does no argument parsing of its own
beyond picking the command - a wrapper that reinterprets flags is a second
place for them to drift.
"""
import sys

COMMANDS = {
    "serve":   ("serve",       "start one vLLM server and wait for it"),
    "bench":   ("bench",       "measure the server that is running"),
    "sweep":   ("runner",      "measure many configurations, with the "
                               "reference control"),
    "fit":     ("costmodel",   "learn from the ledger and validate the fit"),
    "plan":    ("planner",     "recommend a configuration for your limits"),
    "quality": ("quality",     "measure what a precision costs in quality"),
    "watch":   ("gpuwatch",    "sample the GPU during a run"),
    "merge":   ("mergeledger", "combine ledgers from several sessions"),
    "repair":  ("repairledger", "re-pair ledger rows with their own "
                                "benchmark JSON (finding #40)"),
}

ORDER = ["serve", "bench", "sweep", "fit", "plan", "quality", "watch",
         "merge", "repair"]


def usage(stream=sys.stdout) -> None:
    print("inferplan <command> [options]", file=stream)
    print(file=stream)
    print("  a typical first session:", file=stream)
    print("    inferplan sweep --sweep knee --reference-every 4", file=stream)
    print("    inferplan fit", file=stream)
    print("    inferplan plan --rate 20 --p99-ms 2000", file=stream)
    print(file=stream)
    print("  commands:", file=stream)
    for name in ORDER:
        _, blurb = COMMANDS[name]
        print(f"    {name:<9} {blurb}", file=stream)
    print(file=stream)
    print("  any command's own --help works:  inferplan plan --help",
          file=stream)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        usage()
        return 0
    if argv[0] in ("-V", "--version", "version"):
        from . import __version__
        print(f"inferplan {__version__}")
        return 0

    cmd = argv[0]
    if cmd not in COMMANDS:
        print(f"inferplan: unknown command {cmd!r}", file=sys.stderr)
        print(file=sys.stderr)
        usage(sys.stderr)
        return 2

    module_name, _ = COMMANDS[cmd]
    # argparse inside each module reads sys.argv, so the command word is
    # removed and prog is set to how the user actually invoked it. Without
    # that, every error message and --help would say "cli.py".
    sys.argv = [f"inferplan {cmd}"] + argv[1:]
    mod = __import__(f"inferplan.{module_name}", fromlist=["main"])
    rc = mod.main()
    return 0 if rc is None else rc


if __name__ == "__main__":
    raise SystemExit(main())
