"""Quick decode-tok/s benchmark to compare MTP vs vanilla speculative decoding.

Boots a fresh llama-server with the requested binary + model, sends a
fixed prompt via the OpenAI-compatible streaming endpoint, measures
tokens/sec, then tears the server down. Repeat for each (binary, model,
extra-args) combo and print a side-by-side table.

Usage (after the MTP GGUF is downloaded):
    python benchmarks/mtp_bench.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

PROMPT = (
    "Write a Python function `is_prime(n)` that returns True if n is prime "
    "and False otherwise. Then test it on the first 30 integers and print "
    "the primes. Show the full code and expected output. Be thorough, "
    "include a docstring and inline comments."
)
MAX_TOKENS = 400
N_RUNS = 3
PORT = 8090   # avoid colliding with the running AINow llama-server on 8080
LLAMA_HOST = "http://127.0.0.1:" + str(PORT)


def _wait_ready(deadline_s: float = 240.0) -> bool:
    start = time.monotonic()
    while time.monotonic() - start < deadline_s:
        try:
            r = httpx.get(LLAMA_HOST + "/v1/models", timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _stream_decode(model_id: str) -> tuple[int, float]:
    """Returns (tokens, elapsed_s) for one streaming completion."""
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": True,
    }
    n_tok = 0
    t0 = None
    with httpx.stream(
        "POST", LLAMA_HOST + "/v1/chat/completions",
        json=body, timeout=120.0,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                d = json.loads(line[6:])
            except Exception:
                continue
            choices = d.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("content"):
                if t0 is None:
                    t0 = time.monotonic()
                n_tok += 1
    if t0 is None:
        return 0, 0.0
    return n_tok, time.monotonic() - t0


def _run_one(label: str, exe: str, model: str, ngl: int, ctx: int, extra: list[str]):
    args = [
        exe, "-m", model,
        "--port", str(PORT), "--host", "127.0.0.1",
        "-c", str(ctx), "-ngl", str(ngl),
        "-ctk", "q4_0", "-ctv", "q4_0",
        "--no-warmup",
    ] + extra
    print(f"\n=== {label} ===")
    print("cmd:", " ".join('"' + a + '"' if " " in a else a for a in args))
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not _wait_ready():
            print("  llama-server failed to become ready within 90s")
            return
        # Discard a warmup run
        try:
            _stream_decode("any")
        except Exception:
            pass
        toks_per_s = []
        for i in range(N_RUNS):
            n, dt = _stream_decode("any")
            tps = n / dt if dt > 0 else 0
            print(f"  run {i+1}: {n} tok in {dt:.2f}s = {tps:.1f} tok/s")
            toks_per_s.append(tps)
        if toks_per_s:
            avg = sum(toks_per_s) / len(toks_per_s)
            print(f"  -> avg: {avg:.1f} tok/s")
            return avg
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(2)
    return None


def main():
    repo = Path(__file__).resolve().parent.parent
    tq_exe = repo / "llama-server-turboquant" / "llama-server.exe"
    mtp_exe = repo / "llama-server-mtp" / "llama-server.exe"
    base_model = Path(os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf"))
    if not base_model.is_file():
        # Fall back to whatever 4B GGUF the user has (skip MTP variants)
        for cand in Path(os.path.expanduser("~/.lmstudio/models")).rglob("*Qwen3.5-4B*Q4_K_M*.gguf"):
            if "MTP" not in cand.name:
                base_model = cand
                break
    mtp_model = Path(os.path.expanduser("~/.lmstudio/models/localweights/Qwen3.5-4B-MTP-Q4_K_M-GGUF/Qwen3.5-4B-MTP-Q4_K_M.gguf"))

    print(f"baseline model: {base_model}")
    print(f"MTP model:      {mtp_model}")
    print(f"prompt: {PROMPT!r}")
    print(f"max_tokens: {MAX_TOKENS}, n_runs: {N_RUNS}")

    results = {}
    if base_model.is_file() and tq_exe.is_file():
        results["baseline (TurboQuant binary, vanilla model)"] = _run_one(
            "baseline (TurboQuant)", str(tq_exe), str(base_model), 99, 4096, []
        )
    if mtp_model.is_file() and mtp_exe.is_file():
        # MTP requires `--parallel 1` (the binary refuses to start with the
        # default n_parallel=4). The same constraint holds when running via
        # AINow — we'll need to pass --parallel 1 from model_manager.
        results["MTP (PR #22673 binary + MTP-equipped model)"] = _run_one(
            "MTP (PR #22673)", str(mtp_exe), str(mtp_model), 99, 4096,
            ["--spec-type", "mtp", "--spec-draft-n-max", "3", "--parallel", "1"],
        )
        # Also bench the same MTP binary WITHOUT MTP for a within-binary
        # comparison (isolates the MTP feature impact from binary build
        # differences).
        results["MTP binary, MTP off (same binary, vanilla model)"] = _run_one(
            "MTP binary, no spec", str(mtp_exe), str(base_model), 99, 4096, []
        )

    print("\n\n=== summary ===")
    for k, v in results.items():
        print(f"  {k:<55} {v if v is not None else 'failed':>10}")
    if "baseline (TurboQuant binary, vanilla model)" in results and "MTP (PR #22673 binary + MTP-equipped model)" in results:
        b = results["baseline (TurboQuant binary, vanilla model)"]
        m = results["MTP (PR #22673 binary + MTP-equipped model)"]
        if b and m:
            print(f"\nMTP speedup over baseline: {m/b:.2f}x")


if __name__ == "__main__":
    main()
