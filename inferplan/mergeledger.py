#!/usr/bin/env python3
"""
Merge the trustworthy runs from several ledgers into one.

Why this exists: the 23 Nov contaminated ledger holds BOTH good and bad runs.
The prefix-curve sweep of 20 Nov was clean (monotone on three metrics, noise
0.1-1.8%); the knee sweep of 23 Nov was measured next to an orphan server and
is wrong by up to 7x. Throwing the whole file away would lose the prefill data
the cost model needs; keeping it would poison the fit.

So runs are selected by sweep name, newest copy of each label wins, and the
result is audited before it is written.

Usage
    python mergeledger.py \
        --keep ledger.jsonl \
        --take ledger_contaminated_23nov.jsonl --sweeps prefix,prefix_curve \
        --out ledger_clean.jsonl
"""
import argparse
import json
from pathlib import Path


def read(path: Path) -> list:
    rows = []
    if not path.exists():
        print(f"  {path} not found, skipping")
        return rows
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def describe(rows: list) -> dict:
    by_sweep = {}
    for r in rows:
        by_sweep.setdefault(r.get("sweep", "?"), []).append(r)
    return by_sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="append", default=[],
                    help="ledger to keep entirely (repeatable)")
    ap.add_argument("--take", action="append", default=[],
                    help="ledger to take only selected sweeps from")
    ap.add_argument("--sweeps", default="",
                    help="comma-separated sweep names to take from --take files")
    ap.add_argument("--out", default="ledger_clean.jsonl")
    ap.add_argument("--max-noise-pct", type=float, default=25.0)
    args = ap.parse_args()

    wanted = {s.strip() for s in args.sweeps.split(",") if s.strip()}
    merged = {}          # label -> row, newest wins
    kept_from = {}

    for path in args.keep:
        rows = read(Path(path))
        print(f"{path}: {len(rows)} run(s)")
        for sweep, rs in sorted(describe(rows).items()):
            print(f"    sweep {sweep:<14} {len(rs)} run(s)   [keeping all]")
        for r in rows:
            merged[r.get("label")] = r
            kept_from[r.get("label")] = path

    for path in args.take:
        rows = read(Path(path))
        print(f"{path}: {len(rows)} run(s)")
        for sweep, rs in sorted(describe(rows).items()):
            mark = "taking" if sweep in wanted else "DROPPING"
            print(f"    sweep {sweep:<14} {len(rs)} run(s)   [{mark}]")
        for r in rows:
            if r.get("sweep") not in wanted:
                continue
            label = r.get("label")
            # A label already present from a --keep file is newer and wins.
            if label in merged and kept_from.get(label) in args.keep:
                print(f"    skip {label[:46]}: already present from "
                      f"{kept_from[label]}")
                continue
            merged[label] = r
            kept_from[label] = path

    rows = list(merged.values())

    noisy = [r for r in rows if float(r.get("noise_pct") or 0) > args.max_noise_pct]
    if noisy:
        print()
        print(f"  {len(noisy)} merged run(s) still exceed "
              f"{args.max_noise_pct:.0f}% internal noise:")
        for r in noisy:
            print(f"    {r.get('label', '?')[:50]}  "
                  f"{float(r.get('noise_pct') or 0):.0f}%")
        print("  costmodel.py will exclude these from the fit.")

    out = Path(args.out)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    print()
    print(f"wrote {len(rows)} run(s) to {out}")
    for sweep, rs in sorted(describe(rows).items()):
        print(f"    sweep {sweep:<14} {len(rs)} run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
