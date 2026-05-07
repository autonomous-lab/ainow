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
            # Count any content-bearing field. Qwen 3.6 emits
            # `reasoning_content` instead of `content` when thinking
            # is enabled — same compute cost per chunk.
            chunk = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
            if chunk:
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
        # Disable reasoning so we measure pure decode speed (Qwen 3.6
        # defaults to thinking-on which adds variance across runs).
        "--reasoning", "off", "--reasoning-budget", "0",
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


def _bench_model_size(label: str, base_model: Path, mtp_model: Path,
                      tq_exe: Path, mtp_exe: Path, ngl: int, ctx: int) -> dict:
    """Run the 3-setup comparison on a single model size."""
    print(f"\n\n########## {label} ##########")
    print(f"baseline: {base_model}")
    print(f"MTP:      {mtp_model}")
    results = {}
    if base_model.is_file() and tq_exe.is_file():
        results[f"{label}: baseline (TurboQuant binary)"] = _run_one(
            f"{label} baseline", str(tq_exe), str(base_model), ngl, ctx, []
        )
    if mtp_model.is_file() and mtp_exe.is_file():
        results[f"{label}: MTP binary, MTP off"] = _run_one(
            f"{label} MTP-off", str(mtp_exe), str(base_model), ngl, ctx, []
        )
        results[f"{label}: MTP binary + MTP on"] = _run_one(
            f"{label} MTP-on", str(mtp_exe), str(mtp_model), ngl, ctx,
            ["--spec-type", "mtp", "--spec-draft-n-max", "3", "--parallel", "1"],
        )
    return results


def main():
    repo = Path(__file__).resolve().parent.parent
    tq_exe = repo / "llama-server-turboquant" / "llama-server.exe"
    mtp_exe = repo / "llama-server-mtp" / "llama-server.exe"
    models = Path(os.path.expanduser("~/.lmstudio/models"))

    print(f"prompt: {PROMPT!r}")
    print(f"max_tokens: {MAX_TOKENS}, n_runs: {N_RUNS}")

    all_results = {}

    # 4B
    b4 = models / "lmstudio-community/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf"
    m4 = models / "localweights/Qwen3.5-4B-MTP-Q4_K_M-GGUF/Qwen3.5-4B-MTP-Q4_K_M.gguf"
    if b4.is_file() and m4.is_file():
        all_results.update(_bench_model_size("Qwen 3.5 4B", b4, m4, tq_exe, mtp_exe, 99, 4096))

    # 27B (dense). Note: baseline IQ2_M and MTP IQ4_XS are different
    # quants — the MTP variant is at higher precision so without MTP
    # it would normally be slower per-token. Any speedup on the MTP-on
    # row is the feature winning even against that quant disadvantage.
    b27 = models / "unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-IQ2_M.gguf"
    m27 = models / "localweights/Qwen3.6-27B-MTP-IQ4_XS-GGUF/Qwen3.6-27B-MTP-IQ4_XS.gguf"
    if b27.is_file() and m27.is_file():
        all_results.update(_bench_model_size("Qwen 3.6 27B", b27, m27, tq_exe, mtp_exe, 99, 4096))

    print("\n\n=== summary ===")
    for k, v in all_results.items():
        v_str = f"{v:.1f} tok/s" if v is not None else "failed"
        print(f"  {k:<55} {v_str:>15}")


if __name__ == "__main__":
    main()
