#!/usr/bin/env python3
"""
Repair a ledger whose rows were paired with the wrong benchmark JSON.

Finding #40: runner.py located each run's metrics with
glob(f"{label}_*.json"), and every label is a prefix of longer ones -
"..._kvauto" is a prefix of "..._kvauto_nopc" and "..._kvauto_sp512_nopc".
Because "s" sorts after "2", the lexicographically last match was often a
DIFFERENT configuration's file. Any run whose label is a strict prefix of
another label that already had a JSON on disk therefore recorded another
run's numbers.

The benchmark's own JSON files are correct, and so is results.csv. Only the
association in ledger.jsonl is wrong, which means this is repairable without
re-measuring anything: for each ledger row, find the JSON whose name is
exactly "<label>_<stamp>.json" and whose stamp is closest to that row's
timestamp, and put its numbers back.

Usage
    python repairledger.py                       # report what would change
    python repairledger.py --write                # repair in place (backs up)
    python repairledger.py --ledger other.jsonl --runs-dir runs
"""
import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

STAMP = re.compile(r"_(\d{8}T\d{6}Z)\.json$")


def stamp_to_dt(s: str):
    try:
        return datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc)
    except ValueError:
        return None


def row_time(row: dict):
    ts = row.get("timestamp")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def index_runs(runs_dir: Path) -> dict:
    """label -> [(datetime, path)], from exactly-named files only."""
    out = {}
    if not runs_dir.is_dir():
        return out
    for f in sorted(runs_dir.glob("*.json")):
        m = STAMP.search(f.name)
        if not m:
            continue
        label = f.name[:m.start()]
        when = stamp_to_dt(m.group(1))
        if when:
            out.setdefault(label, []).append((when, f))
    for v in out.values():
        v.sort()
    return out


def pick(candidates: list, when):
    """The file whose stamp is closest to the row's timestamp."""
    if not candidates:
        return None
    if when is None:
        return candidates[-1][1]
    return min(candidates, key=lambda c: abs((c[0] - when).total_seconds()))[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="ledger.jsonl")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--write", action="store_true",
                    help="apply the repair (the original is backed up first)")
    ap.add_argument("--metric", default="e2e_p50",
                    help="which metric to show in the report")
    args = ap.parse_args()

    ledger = Path(args.ledger)
    if not ledger.exists():
        print(f"  {ledger} not found")
        return 1
    index = index_runs(Path(args.runs_dir))
    if not index:
        print(f"  no exactly-named JSON files in {args.runs_dir}; nothing to "
              f"repair from")
        return 1

    rows, changed, missing, unchanged = [], [], [], 0
    gained_length = 0
    for line in ledger.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            rows.append(line)              # kept verbatim
            continue
        label = row.get("label")
        if not label or row.get("kind") == "reference" \
                or row.get("outcome") == "CRASH":
            rows.append(row)
            continue
        path = pick(index.get(label, []), row_time(row))
        if path is None:
            missing.append(label)
            rows.append(row)
            continue
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            missing.append(label)
            rows.append(row)
            continue
        before = (row.get("median") or {}).get(args.metric)
        after = (rec.get("median") or {}).get(args.metric)
        median = dict(rec.get("median") or {})
        # Older JSON files predate out_tokens_mean as a reported metric, but
        # they all carry per-pass out_tokens and requests_ok, so the mean can
        # be recovered rather than re-measured. Finding #41: without it every
        # prediction assumes max_tokens and runs 9-25% high.
        if "out_tokens_mean" not in median:
            passes = rec.get("passes") or []
            tot = sum(float(ps.get("out_tokens") or 0) for ps in passes)
            ok = sum(float(ps.get("requests_ok") or 0) for ps in passes)
            if ok:
                median["out_tokens_mean"] = tot / ok
        row["median"] = median
        row["noise_pct"] = rec.get("noise_pct")
        row["status"] = rec.get("status")
        row["bench_json"] = str(path)
        row["repaired_from"] = path.name
        if "out_tokens_mean" in median:
            gained_length += 1
        if before is not None and after is not None \
                and abs(before - after) > 1e-6:
            changed.append((label, before, after))
        else:
            unchanged += 1
        rows.append(row)

    print("=" * 78)
    print(f"  LEDGER REPAIR  -  {ledger}")
    print("=" * 78)
    print(f"  rows matched to their own JSON : {unchanged + len(changed)}")
    print(f"  rows whose numbers CHANGED     : {len(changed)}")
    print(f"  rows with a completion length  : {gained_length} "
          f"(recovered from per-pass counts where missing)")
    if missing:
        print(f"  rows with no exact JSON        : {len(missing)} "
              f"(left untouched)")
        for lb in sorted(set(missing))[:5]:
            print(f"      {lb[-50:]}")
    if changed:
        print("-" * 78)
        print(f"  {'run':<48}{'was':>9}{'now':>9}")
        for label, before, after in changed:
            short = label if len(label) <= 47 else "..." + label[-44:]
            print(f"  {short:<48}{before:>9.1f}{after:>9.1f}")
    print("=" * 78)

    if not args.write:
        print("  nothing written. Re-run with --write to apply.")
        return 0
    if not changed and not gained_length:
        print("  nothing to change.")
        return 0

    backup = ledger.with_suffix(
        ledger.suffix + "." +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".bak")
    shutil.copy2(ledger, backup)
    with open(ledger, "w") as f:
        for row in rows:
            f.write((row if isinstance(row, str) else json.dumps(row)) + "\n")
    print(f"  repaired {len(changed)} row(s); {gained_length} row(s) now "
          f"carry a completion length.")
    print(f"  original saved as {backup}")
    print(f"  now re-run:  inferplan fit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())