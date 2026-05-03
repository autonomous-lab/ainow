"""Aider Polyglot benchmark harness for AINow.

Adapted from itayinbarr/little-coder so the numbers are directly comparable
(same prompt template, same retry/scoring rules, same JSON schema).

Difference: instead of spawning `pi --mode rpc`, we spawn AINow's CLI in
one-shot mode (`python -m src.cli --yolo --no-banner --no-textual`) inside
a fresh per-exercise workspace.

Usage:
    python benchmarks/aider_polyglot.py --model online --exercises 5
    python benchmarks/aider_polyglot.py --model 9b --resume

Dataset: https://github.com/Aider-AI/polyglot-benchmark
    git clone https://github.com/Aider-AI/polyglot-benchmark <root>
    # only Python is wired up here for now (matches little-coder v0)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = Path(os.getenv("POLYGLOT_BENCHMARK_ROOT", "D:/dev/polyglot-benchmark"))
RESULTS_PATH = REPO_ROOT / "benchmarks" / "results_polyglot.json"
LOG_DIR = REPO_ROOT / "benchmarks" / "polyglot_logs"


PROMPT = (
    "Implement the Exercism exercise `{name}`.\n\n"
    "Stub file(s) to implement:\n{stubs}\n\n"
    "Test file(s) (for reference only — DO NOT edit):\n{tests}\n\n"
    "{hint}\n\n"
    "Read the stubs + any `.docs/instructions.md` in the workspace, "
    "then implement the solution. When you believe the code is correct, "
    "stop calling tools."
)

RETRY_PROMPT = (
    "The tests failed. Output:\n\n```\n{tail}\n```\n\n"
    "Fix the implementation and try again."
)


def _copy_exercise(src: Path, dst: Path) -> Path:
    # On Windows, lingering pytest/unittest processes from a prior run can
    # hold the dir open. Retry with a short backoff; if still stuck, fall
    # back to a fresh sibling dir so the harness keeps moving. Returns the
    # actual destination path (may differ from `dst` if rmtree gave up).
    if dst.exists():
        for attempt in range(5):
            try:
                shutil.rmtree(dst)
                break
            except PermissionError:
                if attempt == 4:
                    dst = dst.parent / f"{dst.name}.r{int(time.time())}"
                    break
                time.sleep(1.0 + attempt)
    shutil.copytree(src, dst)
    return dst


def _run_cmd(cmd, work: Path, timeout: int, shell: bool = False):
    try:
        r = subprocess.run(
            cmd, cwd=str(work),
            capture_output=True, text=True, timeout=timeout, shell=shell,
        )
        return r.returncode == 0, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    except FileNotFoundError as e:
        return False, f"toolchain missing: {e}"


# --- Python ---
def _prepare_python(work: Path):
    stubs = [p for p in work.glob("*.py") if not p.name.endswith("_test.py")]
    tests = list(work.glob("*_test.py"))
    return stubs, tests


def _run_python_tests(work: Path, timeout: int):
    return _run_cmd([sys.executable, "-m", "pytest", "-x", "-q"], work, timeout)


# --- JavaScript ---
def _prepare_javascript(work: Path):
    stubs = [
        p for p in work.glob("*.js")
        if not p.name.endswith(".spec.js") and p.name not in {"babel.config.js"}
    ]
    tests = list(work.glob("*.spec.js"))
    return stubs, tests


def _run_javascript_tests(work: Path, timeout: int):
    # `npm install` first so jest/babel resolve. Most exercises ship a
    # package.json with `npm test` wired up to jest.
    ok, out = _run_cmd(["npm", "install", "--silent", "--no-audit", "--no-fund"], work, 120, shell=True)
    if not ok:
        return False, "npm install failed:\n" + out
    return _run_cmd(["npm", "test", "--silent"], work, timeout, shell=True)


# --- Go ---
def _prepare_go(work: Path):
    stubs = [
        p for p in work.glob("*.go")
        if not p.name.endswith("_test.go")
    ]
    tests = list(work.glob("*_test.go"))
    return stubs, tests


def _run_go_tests(work: Path, timeout: int):
    return _run_cmd(["go", "test", "./..."], work, timeout, shell=True)


# --- Rust ---
def _prepare_rust(work: Path):
    src = work / "src"
    tests = work / "tests"
    stubs = list(src.glob("*.rs")) if src.is_dir() else []
    test_files = list(tests.glob("*.rs")) if tests.is_dir() else []
    return stubs, test_files


def _run_rust_tests(work: Path, timeout: int):
    return _run_cmd(["cargo", "test", "--", "--include-ignored"], work, timeout, shell=True)


# --- C++ ---
def _prepare_cpp(work: Path):
    stubs = [
        p for p in list(work.glob("*.cpp")) + list(work.glob("*.h"))
        if not p.name.endswith("_test.cpp")
    ]
    tests = list(work.glob("*_test.cpp"))
    return stubs, tests


def _run_cpp_tests(work: Path, timeout: int):
    build = work / "build"
    build.mkdir(exist_ok=True)
    ok, out = _run_cmd(["cmake", "-B", "build", "-S", "."], work, 60, shell=True)
    if not ok:
        return False, "cmake configure failed:\n" + out
    ok, out = _run_cmd(["cmake", "--build", "build", "--config", "Release"], work, 120, shell=True)
    if not ok:
        return False, "cmake build failed:\n" + out
    return _run_cmd(["ctest", "--test-dir", "build", "--output-on-failure"], work, timeout, shell=True)


# --- Java ---
def _prepare_java(work: Path):
    main = work / "src" / "main" / "java"
    test = work / "src" / "test" / "java"
    stubs = list(main.rglob("*.java")) if main.is_dir() else []
    tests = list(test.rglob("*.java")) if test.is_dir() else []
    return stubs, tests


def _run_java_tests(work: Path, timeout: int):
    gradlew = "gradlew.bat" if os.name == "nt" else "./gradlew"
    return _run_cmd([gradlew, "test"], work, timeout, shell=True)


LANGS = {
    "python": {
        "practice_dir": "python/exercises/practice",
        "prepare": _prepare_python,
        "run_tests": _run_python_tests,
        "hint": "Use Python 3. Run tests with `python -m pytest -x -q`.",
        "test_timeout": 90,
    },
    "javascript": {
        "practice_dir": "javascript/exercises/practice",
        "prepare": _prepare_javascript,
        "run_tests": _run_javascript_tests,
        "hint": "Use ES modules / modern JS. Run tests with `npm test` (jest).",
        "test_timeout": 180,
    },
    "go": {
        "practice_dir": "go/exercises/practice",
        "prepare": _prepare_go,
        "run_tests": _run_go_tests,
        "hint": "Run tests with `go test ./...`.",
        "test_timeout": 180,
    },
    "rust": {
        "practice_dir": "rust/exercises/practice",
        "prepare": _prepare_rust,
        "run_tests": _run_rust_tests,
        "hint": "Edit `src/lib.rs`. Run tests with `cargo test -- --include-ignored`.",
        "test_timeout": 240,
    },
    "cpp": {
        "practice_dir": "cpp/exercises/practice",
        "prepare": _prepare_cpp,
        "run_tests": _run_cpp_tests,
        "hint": "Edit `<exercise>.cpp` / `.h`. Build + test with CMake + ctest.",
        "test_timeout": 180,
    },
    "java": {
        "practice_dir": "java/exercises/practice",
        "prepare": _prepare_java,
        "run_tests": _run_java_tests,
        "hint": "Edit `src/main/java/...`. Run tests with `./gradlew test`.",
        "test_timeout": 240,
    },
}


def _build_prompt(name: str, stubs, tests, hint: str) -> str:
    return PROMPT.format(
        name=name,
        stubs="\n".join(f"  - {p.name}" for p in stubs),
        tests="\n".join(f"  - {p.name}" for p in tests),
        hint=hint,
    )


def _run_ainow(prompt: str, cwd: Path, model: str, log_path: Path, timeout: int) -> int:
    """Spawn AINow CLI in one-shot mode, stream output to log_path."""
    cmd = [
        sys.executable, "-m", "src.cli",
        "--no-banner", "--no-textual", "--yolo",
        "-m", model,
        "-a", "default",
        "--cwd", str(cwd),
        prompt,
    ]
    env = dict(os.environ)
    # cwd is the exercise dir (so file/glob/bash tools land there) but
    # `src.cli` lives in REPO_ROOT — add it to PYTHONPATH so -m src.cli works.
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["TERM"] = "dumb"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as logf:
        try:
            # Run from REPO_ROOT so agents/ resolves to the AINow agents
            # store (not a fresh empty one in the exercise). The tool
            # sandbox is steered to the exercise via --cwd above.
            proc = subprocess.run(
                cmd, cwd=str(REPO_ROOT),
                stdout=logf, stderr=subprocess.STDOUT,
                env=env, timeout=timeout,
            )
            return proc.returncode
        except subprocess.TimeoutExpired:
            logf.write(f"\n[harness] timed out after {timeout}s\n".encode())
            return -1


def _load_results() -> dict:
    if RESULTS_PATH.exists():
        try:
            return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_results(d: dict) -> None:
    RESULTS_PATH.write_text(json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="online", help="AINow model alias (e.g. 9b, online, online2)")
    p.add_argument("--lang", default="python", choices=list(LANGS.keys()))
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--exercises", type=int, default=0, help="0 = all")
    p.add_argument("--agent-timeout", type=int, default=900, help="per-AINow-call timeout (s)")
    p.add_argument("--resume", action="store_true", help="skip exercises already in results JSON")
    p.add_argument("--no-retry", action="store_true", help="do not retry on first failure")
    p.add_argument("--only", nargs="*", help="run only these exercise names")
    args = p.parse_args(argv)

    desc = LANGS[args.lang]
    practice = args.dataset / desc["practice_dir"]
    if not practice.exists():
        print(f"dataset not found: {practice}", file=sys.stderr)
        return 1

    exercises = sorted([p for p in practice.iterdir() if p.is_dir()])
    if args.only:
        wanted = set(args.only)
        exercises = [e for e in exercises if e.name in wanted]
    if args.exercises:
        exercises = exercises[: args.exercises]

    # Always preserve prior results; --resume skips re-running them, --only
    # forces a re-run of the named ones (so we can iterate on fixes).
    results = _load_results()
    run_tag = f"{args.model}/{args.lang}"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    work_root = REPO_ROOT / "benchmarks" / "_work" / args.lang
    work_root.mkdir(parents=True, exist_ok=True)

    for i, ex_src in enumerate(exercises, 1):
        name = ex_src.name
        key = f"{run_tag}/{name}"
        # --only forces a re-run; otherwise --resume skips anything that
        # already has a recorded status.
        if not args.only and args.resume and key in results and results[key].get("status") in {"pass_1", "pass_2", "fail"}:
            print(f"[{i}/{len(exercises)}] skip (resume): {name} -> {results[key]['status']}")
            continue

        work = work_root / name
        work = _copy_exercise(ex_src, work)
        stubs, tests = desc["prepare"](work)
        if not stubs or not tests:
            results[key] = {"status": "skipped", "reason": "no stubs/tests"}
            _save_results(results)
            continue

        prompt = _build_prompt(name, stubs, tests, desc["hint"])
        log_run = LOG_DIR / f"{args.model.replace('/', '_')}_{name}.log"

        t0 = time.monotonic()
        print(f"[{i}/{len(exercises)}] {name} ...", flush=True)
        rc = _run_ainow(prompt, work, args.model, log_run, args.agent_timeout)
        passed, out = desc["run_tests"](work, desc["test_timeout"])
        status = "pass_1" if passed else None

        if not passed and not args.no_retry:
            retry = RETRY_PROMPT.format(tail=out[-4000:])
            log_run2 = LOG_DIR / f"{args.model.replace('/', '_')}_{name}.retry.log"
            rc2 = _run_ainow(retry, work, args.model, log_run2, args.agent_timeout)
            passed, out = desc["run_tests"](work, desc["test_timeout"])
            if passed:
                status = "pass_2"

        if status is None:
            status = "fail"

        elapsed = time.monotonic() - t0
        results[key] = {
            "status": status,
            "elapsed_s": round(elapsed, 1),
            "agent_rc": rc,
        }
        _save_results(results)
        print(f"  -> {status}  ({elapsed:.1f}s)", flush=True)

    # Aggregate
    bucket = {k: v for k, v in results.items() if k.startswith(run_tag + "/")}
    n_pass = sum(1 for v in bucket.values() if v.get("status", "").startswith("pass"))
    n_total = len(bucket)
    print(f"\n{run_tag}: {n_pass}/{n_total} passed ({100 * n_pass / max(1, n_total):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
