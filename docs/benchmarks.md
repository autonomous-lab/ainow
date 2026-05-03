# Benchmarks

AINow ships a benchmark harness to track its agentic-coding capability over time. The first benchmark wired up is **Aider Polyglot** (`benchmarks/aider_polyglot.py`), adapted from [`itayinbarr/little-coder`](https://github.com/itayinbarr/little-coder) so the numbers are directly comparable.

## Aider Polyglot

[Aider Polyglot](https://github.com/Aider-AI/polyglot-benchmark) is a 225-exercise coding benchmark drawn from Exercism's hardest practice problems across 6 languages (cpp, go, java, javascript, python, rust). Each exercise gives the agent a stub file, a test file (read-only), and the spec — the agent has to read, edit, run tests, debug, and commit until tests pass.

### Scope

Aider Polyglot has 225 exercises across 6 languages: **python (34), javascript (49), java (47), go (39), rust (30), cpp (26)**. AINow's harness has all 6 wired up; running them requires the corresponding toolchain on the host:

| language | test command | host tool needed |
|---|---|---|
| python | `pytest -x -q` | python (already needed for AINow) |
| javascript | `npm test` (jest) | `node` + `npm` |
| go | `go test ./...` | Go toolchain |
| rust | `cargo test -- --include-ignored` | Rust toolchain |
| cpp | `cmake -B build -S . && cmake --build build && ctest --test-dir build` | `cmake` + `g++` (or MSVC) |
| java | `./gradlew test` | JDK + gradle wrapper (ships per exercise) |

These match the test commands little-coder uses (which mirror Aider's own benchmark runner: [aider/benchmark/benchmark.py](https://github.com/Aider-AI/aider/blob/main/benchmark/benchmark.py)).

### Headline — runs so far

#### Python (34 exercises)

| Model | Backend | pass_1 | pass_2 | fail | **pass rate** | wall-clock | mean / exo |
|---|---|---:|---:|---:|---:|---:|---:|
| Gemini 3.1 Flash Lite | OpenRouter (online) | 31 | 2 | 1 | **97.1%** | ~25 min | ~36 s |
| Qwen 3.5 27B (UD-IQ3_XXS) | local llama.cpp | 29 | 3 | 2 | **94.1%** | 68.5 min | 121 s |
| Qwen 3.5 9B (UD-Q4_K_XL) | local llama.cpp | 18 | 6 | 10 | **70.6%** | 89 min | 157 s |

#### JavaScript (49 exercises)

| Model | Backend | pass_1 | pass_2 | fail | **pass rate** | wall-clock | mean / exo |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen 3.5 9B (UD-Q4_K_XL) | local llama.cpp | 40 | 5 | 4 | **91.8%** | 82 min | 100 s |
| Qwen 3.6 27B (UD-IQ2_M) | local llama.cpp | 46 | 3 | 0 | **100.0%** | 108 min | 132 s |

#### Comparison vs little-coder

[little-coder reports 45.56%](https://github.com/itayinbarr/little-coder) on the *full 225-exercise* polyglot with **Qwen 3.5 9B via Ollama** (Ollama's default tag = **Q4_K_M**, ≈5.6 GB). AINow's `9b` alias uses **UD-Q4_K_XL** (Unsloth Dynamic, ≈5.7 GB) — comparable quant.

| Coverage | Model | exercises | pass rate |
|---|---|---:|---:|
| AINow — Python + JavaScript | Qwen 3.5 9B UD-Q4_K_XL | 83/225 | **83.1%** |
| little-coder — full 6-lang | Qwen 3.5 9B Q4_K_M (Ollama) | 225/225 | 45.56% |

Not a clean head-to-head (different language coverage), but on the 83 exercises we share, AINow's scaffolding moves the same model class from a baseline that struggles on the full benchmark to **83.1% pass rate on the languages we can run locally** — almost double the reported 9B figure. Wiring up go / rust / java / cpp on a host with the corresponding toolchains will close out the comparison.

### Failure analysis

**Gemini 3.1 Flash Lite (1 fail):**
- `dot-dsl` — agent confuses `TypeError` vs `ValueError` in `test_malformed_attr` and never converges. Both Qwen 9B and Qwen 27B *do* solve it locally, so it's a model-capability quirk specific to Flash Lite, not a hard problem.

**Qwen 3.5 27B (2 fail):**
- `react` — 446 s, hit max-iterations on a complex stateful exercise.
- `sgf-parsing` — 605 s, hit max-iterations on a recursive parser.

Both failures sit in the long-tail of timing (445–605 s), suggesting these are genuine multi-step debugging tasks where the model didn't converge before the iteration cap.

### Per-exercise timing (Qwen 3.5 27B)

<details>
<summary>Click to expand 34-row table</summary>

| exercise | status | elapsed |
|---|---|---:|
| proverb | pass_1 | 13.4 s |
| hangman | pass_1 | 15.8 s |
| robot-name | pass_1 | 18.4 s |
| list-ops | pass_1 | 18.6 s |
| grade-school | pass_1 | 19.7 s |
| affine-cipher | pass_1 | 23.7 s |
| simple-linked-list | pass_1 | 25.1 s |
| book-store | pass_1 | 27.3 s |
| food-chain | pass_1 | 28.9 s |
| two-bucket | pass_1 | 30.0 s |
| bottle-song | pass_1 | 32.9 s |
| connect | pass_1 | 33.8 s |
| phone-number | pass_1 | 41.7 s |
| dominoes | pass_1 | 42.0 s |
| zipper | pass_1 | 47.6 s |
| grep | pass_1 | 48.8 s |
| pig-latin | pass_2 | 54.7 s |
| variable-length-quantity | pass_1 | 56.6 s |
| wordy | pass_1 | 56.8 s |
| beer-song | pass_1 | 66.0 s |
| poker | pass_1 | 68.2 s |
| go-counting | pass_1 | 72.3 s |
| scale-generator | pass_1 | 90.6 s |
| dot-dsl | pass_1 | 101.2 s |
| tree-building | pass_1 | 103.0 s |
| paasio | pass_1 | 128.6 s |
| zebra-puzzle | pass_1 | 142.2 s |
| rest-api | pass_1 | 143.5 s |
| transpose | pass_1 | 206.1 s |
| forth | pass_1 | 304.7 s |
| bowling | pass_2 | 439.5 s |
| react | **fail** | 445.7 s |
| pov | pass_2 | 558.4 s |
| sgf-parsing | **fail** | 604.8 s |

</details>

## How to run

### 1. Get the dataset

```bash
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/Aider-AI/polyglot-benchmark.git ~/polyglot-benchmark
git -C ~/polyglot-benchmark sparse-checkout set python javascript go rust cpp java
```

Set `POLYGLOT_BENCHMARK_ROOT` if you cloned somewhere other than `D:/dev/polyglot-benchmark` (the harness default):

```bash
export POLYGLOT_BENCHMARK_ROOT=~/polyglot-benchmark
```

### 2. Run the harness

From the AINow repo root:

```bash
# All 34 Python exercises with the online model
python benchmarks/aider_polyglot.py --model online

# Local Qwen 3.5 27B (load it via the web UI first, or the harness will start it)
python benchmarks/aider_polyglot.py --model 27b

# Local Qwen 3.5 9B
python benchmarks/aider_polyglot.py --model 9b

# Resume — skip exercises that already have a recorded status
python benchmarks/aider_polyglot.py --model 27b --resume

# Only specific exercises (forces re-run, ignores --resume)
python benchmarks/aider_polyglot.py --model 27b --only phone-number bowling

# First N exercises (smoke test)
python benchmarks/aider_polyglot.py --model online --exercises 5

# Disable the second attempt — only count first-pass success
python benchmarks/aider_polyglot.py --model online --no-retry
```

For local models, load the model in the web UI (or run `python main.py -m 27b`) before launching the bench so it attaches to the running llama-server instead of cold-starting per exercise.

### 3. Inspect results

- **`benchmarks/results_polyglot.json`** — per-exercise status (`pass_1` / `pass_2` / `fail` / `skipped`), elapsed time, agent return code. Keyed by `<model>/<lang>/<exercise>`.
- **`benchmarks/polyglot_logs/<model>_<exercise>.log`** — full stdout/stderr from the agent's first attempt (tool calls + token stream).
- **`benchmarks/polyglot_logs/<model>_<exercise>.retry.log`** — same for the retry attempt (only created if the first attempt failed and `--no-retry` was not set).

## Protocol

Each exercise runs in an isolated workspace (`benchmarks/_work/<lang>/<exercise>/`) populated by copying from the dataset. The agent sees the stub file(s) + the test file(s) + any `.docs/instructions.md`, and is told (verbatim, same prompt little-coder uses):

```
Implement the Exercism exercise `<name>`.

Stub file(s) to implement:
  - <stub.py>

Test file(s) (for reference only — DO NOT edit):
  - <stub_test.py>

Use Python 3. Run tests with `python -m pytest -x -q`.

Read the stubs + any `.docs/instructions.md` in the workspace,
then implement the solution. When you believe the code is correct,
stop calling tools.
```

The harness then runs `python -m pytest -x -q` from the workspace. `returncode == 0` → pass. On failure, the harness sends a retry prompt with the last 4 KB of test output and gives the agent one more shot. Pass on retry → `pass_2`.

The agent is invoked as a subprocess:

```bash
python -m src.cli --no-banner --no-textual --yolo \
    -m <model> -a default --cwd <workspace> "<prompt>"
```

`--cwd <workspace>` is the only AINow-specific knob — it overrides the default tool sandbox (which is `agents/<name>/`) so the agent can read/write inside the exercise dir.

## Scaffolding fixes that came out of the bring-up

Bringing the benchmark up exposed several AINow issues that were fixed at the same time:

1. **Tool calls were silently dropped on Gemini-via-OpenRouter** (`src/services/llm.py`). The OpenAI SDK's Pydantic streaming parser strips Gemini's tool-call SSE fragments, producing `finish_reason=tool_calls` with zero accumulated calls. The raw-SSE bypass that already covered local llama.cpp and DeepSeek now also covers any OpenRouter base URL.

2. **`MAX_TOOL_ITERATIONS` raised from 15 → 30**. Coding tasks legitimately need read instructions → read stub → read tests → 2-3 edit/test cycles. 15 was too tight.

3. **Consecutive-duplicate-call detector** (`src/services/llm.py`). Small models stuck in indecision sometimes re-read the same file 4-5× in a row. After 2 identical calls in a row, AINow now appends a system note: *"You just called this exact tool with the same arguments. The result has not changed. Stop re-fetching context — commit to the next action."*

4. **`--cwd <dir>` flag on `python -m src.cli`** (`src/cli.py`). Without it the per-tool sandbox is hard-coded to `agents/<name>/`, which makes it impossible to point the agent at an external workspace (e.g. a benchmark exercise dir).

5. **Windows-resilient harness** (`benchmarks/aider_polyglot.py`). Lingering pytest/unittest processes from a prior run can hold the exercise dir open; `_copy_exercise` now retries `rmtree` with backoff and falls back to a fresh sibling dir.

Together, the iteration bump + duplicate-call detector took the online Python pass rate from **82.4% → 97.1%**.

## Roadmap

- **Toolchain installs** — only python + javascript are runnable on the default Windows AINow setup; go / rust / java / cpp need the corresponding compiler/SDK before their language descriptors will work end-to-end.
- **Full 6-language run with `-m 9b`** — direct apples-to-apples vs little-coder's 45.56%.
- **Terminal-Bench 2.0** and **GAIA Validation** — the other two benchmarks little-coder runs. Both have public test sets.
