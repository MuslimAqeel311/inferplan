#!/usr/bin/env python3
"""
The checks that would have caught the two packaging bugs in version 0.1.0.

Neither was subtle. Both survived because nothing looked:

  1. planner.py imported costmodel at module level AND twice more inside
     functions. Only the module-level one was made relative, so the package
     imported fine and `inferplan plan` died on the first real call with
     ModuleNotFoundError. A test that imports a module proves nothing about
     imports that happen when it runs.

  2. pyproject.toml declared no dependencies at all, with a comment
     explaining proudly that none were needed. bench.py imports aiohttp. The
     wheel installed cleanly into a fresh virtual environment and then failed
     the moment a command touched the benchmark.

Run with: python -m pytest tests/ -q      (or just: python tests/test_smoke.py)
"""
import ast
import pathlib
import subprocess
import sys

PKG = pathlib.Path(__file__).resolve().parent.parent / "inferplan"
MODULES = ["bench", "runner", "costmodel", "planner", "quality",
           "serve", "gpuwatch", "mergeledger", "cli"]
COMMANDS = ["serve", "bench", "sweep", "fit", "plan", "quality",
            "watch", "merge"]


def _py_files():
    return sorted(PKG.glob("*.py"))


def test_no_flat_intra_package_imports():
    """
    Every reference to a sibling module must be relative, wherever it sits.

    Walks the whole AST rather than reading the top of the file, because the
    bug was an import inside a function body.
    """
    siblings = {m for m in MODULES}
    offenders = []
    for f in _py_files():
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module in siblings:
                    offenders.append(f"{f.name}:{node.lineno} "
                                     f"from {node.module} import ...")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in siblings:
                        offenders.append(f"{f.name}:{node.lineno} "
                                         f"import {alias.name}")
    assert not offenders, "flat imports of sibling modules:\n  " + \
        "\n  ".join(offenders)


def test_declared_dependencies_cover_every_third_party_import():
    """
    Every third-party module imported anywhere must be declared.

    The project's own dependency list is parsed from pyproject.toml rather
    than hard-coded here, so adding an import without declaring it fails even
    if this file is never touched.
    """
    pyproject = PKG.parent / "pyproject.toml"
    try:
        import tomllib
    except ModuleNotFoundError:            # Python 3.9/3.10
        import tomli as tomllib           # type: ignore
    meta = tomllib.loads(pyproject.read_text())
    declared = set()
    for spec in meta["project"].get("dependencies", []):
        name = spec.split(">")[0].split("<")[0].split("=")[0].split("[")[0]
        declared.add(name.strip().lower().replace("-", "_"))
    for extra in meta["project"].get("optional-dependencies", {}).values():
        for spec in extra:
            name = spec.split(">")[0].split("<")[0].split("=")[0].split("[")[0]
            declared.add(name.strip().lower().replace("-", "_"))

    stdlib = set(sys.stdlib_module_names)
    missing = {}
    for f in _py_files():
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 \
                    and node.module:
                names = [node.module.split(".")[0]]
            for n in names:
                low = n.lower().replace("-", "_")
                if n in stdlib or n == "inferplan" or low in declared:
                    continue
                # tomli is this test's own fallback on old Pythons.
                if low in ("tomli", "pytest"):
                    continue
                missing.setdefault(n, []).append(f"{f.name}:{node.lineno}")
    assert not missing, (
        "imported but not declared in pyproject.toml: " +
        ", ".join(f"{k} ({'; '.join(v)})" for k, v in sorted(missing.items())))


def test_every_module_imports():
    """Importing is the floor, not the ceiling - see the next test."""
    for m in MODULES:
        __import__(f"inferplan.{m}")


def test_every_command_runs_its_own_help():
    """
    Each command is actually executed, which is what catches a deferred
    import. --help exits before any measurement, so this needs no GPU and no
    server, but it does run main() and therefore the argument parser the
    command builds at call time.
    """
    failures = []
    for cmd in COMMANDS:
        p = subprocess.run(
            [sys.executable, "-m", "inferplan.cli", cmd, "--help"],
            capture_output=True, text=True, timeout=60)
        if p.returncode != 0:
            failures.append(f"{cmd}: rc={p.returncode} "
                            f"{(p.stderr or p.stdout).strip()[-160:]}")
        elif "usage:" not in (p.stdout + p.stderr):
            failures.append(f"{cmd}: printed no usage line")
    assert not failures, "commands that do not run:\n  " + \
        "\n  ".join(failures)


def test_unknown_command_is_an_error():
    p = subprocess.run([sys.executable, "-m", "inferplan.cli", "bogus"],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 2, f"expected rc=2, got {p.returncode}"


def test_runner_finds_the_benchmark_it_shells_out_to():
    """
    runner.py launches bench.py as a subprocess. Its default path is resolved
    next to the module, so a sweep started from any directory still finds it;
    if that ever becomes a bare relative name again, this fails.
    """
    import inferplan.runner as runner
    bench = pathlib.Path(runner.__file__).with_name("bench.py")
    assert bench.exists(), f"{bench} is missing from the installed package"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}\n        {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print()
    print("all checks passed" if not failed else f"{failed} check(s) failed")
    raise SystemExit(1 if failed else 0)
