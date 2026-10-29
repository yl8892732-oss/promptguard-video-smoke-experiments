# PromptGuard Video Smoke Experiments

This repository is a reviewed delivery of the PromptGuard inference adapter and the first real
small-sample smoke test on CogVideoX-2B. It contains the project source, scripts, tests, the two
benchmark smoke manifests, the frozen smoke protocol, and eight validated video/JSON outputs.

**Scope:** this is a two-benchmark small-sample smoke test plus two control runs. It is **not** a
full benchmark, an official evaluator run, an ablation study, or a paper-level experiment.

## Repository layout

```text
.
├── PromptGuard/
│   ├── promptguard/          # PromptGuard core, semantics, CogVideoX adapter, benchmark loader
│   ├── scripts/              # generation, manifest building, dry-run planner, model download
│   ├── tests/                # CPU / small-model tests and protocol checks
│   ├── pyproject.toml
│   └── requirements.txt
├── artifacts/
│   ├── manifests/            # two 3-record smoke manifests
│   ├── environment/          # smoke_protocol.json, cloud_run_checklist.md
│   ├── sources/              # source records for the two benchmark snapshots
│   ├── dry_runs/             # small Stage 04 plans used by the protocol test
│   └── smoke/stage06_smoke_v1/
│       ├── control/{dense,promptguard}/
│       ├── t2v_compbench/.../
│       ├── vbench/.../
│       └── RUN_REPORT.md
├── EXPERIMENT_REPORT.md
├── README.md
└── .gitignore
```

The code and tests live in `PromptGuard/`, with `artifacts/` as a sibling. This mirrors the original
project layout, which the tests use to locate manifests and protocol files. Run commands from
`PromptGuard/` unless stated otherwise.

## Verified smoke result

The real run used the pinned model revision
`1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01` on an NVIDIA A40 (46,068 MiB, compute capability 8.6),
with `torch 2.7.1+cu126`, `diffusers 0.35.2`, `transformers 4.57.0`, `accelerate 1.15.0`, and
Python 3.11.13.

Frozen settings: seed 42, 8 steps, 9 frames, 720×480, guidance 6.0, density 0.4, alpha 0.2,
delta 0.05, block size 256, warmup 1, offload `model`, VRAM cap 23 GiB.

All eight selected runs completed with exit code 0:

| Run | Output | MP4 bytes | JSON `finite` |
|---|---|---:|---|
| control, dense | `artifacts/smoke/stage06_smoke_v1/control/dense/readme_prompt.mp4` | 8,500 | true |
| control, promptguard | `artifacts/smoke/stage06_smoke_v1/control/promptguard/readme_prompt.mp4` | 27,662 | true |
| T2V-CompBench `consistent_attribute_binding/0001` | `artifacts/smoke/stage06_smoke_v1/t2v_compbench/consistent_attribute_binding/0001.mp4` | 207,253 | true |
| T2V-CompBench `spatial_relationships/0001` | `artifacts/smoke/stage06_smoke_v1/t2v_compbench/spatial_relationships/0001.mp4` | 291,798 | true |
| T2V-CompBench `action_binding/0001` | `artifacts/smoke/stage06_smoke_v1/t2v_compbench/action_binding/0001.mp4` | 259,015 | true |
| VBench `multiple_objects/0076` | `artifacts/smoke/stage06_smoke_v1/vbench/multiple_objects/0076.mp4` | 110,964 | true |
| VBench `human_action/0158` | `artifacts/smoke/stage06_smoke_v1/vbench/human_action/0158.mp4` | 128,120 | true |
| VBench `spatial_relationship/0863` | `artifacts/smoke/stage06_smoke_v1/vbench/spatial_relationship/0863.mp4` | 137,437 | true |

Each MP4 has a same-stem JSON. The JSONs were validated for `finite: true`,
`video_shape: [9, 480, 720, 3]`, seed 42, the frozen generation settings, and peak reserved VRAM
around 10.83 GiB. The MP4s were decoded locally and verified as 720×480, 8 fps, 9 frames,
non-constant content. The detailed per-run report is
[`artifacts/smoke/stage06_smoke_v1/RUN_REPORT.md`](artifacts/smoke/stage06_smoke_v1/RUN_REPORT.md).

The two 3-record manifests are:

- `artifacts/manifests/t2v_compbench_smoke.jsonl`
- `artifacts/manifests/vbench_smoke.jsonl`

T2V-CompBench V1 uses 100 prompts per category; the smoke manifest selects one ID from each of
`consistent_attribute_binding`, `spatial_relationships`, and `action_binding`.

## Environment and setup

The project targets Python ≥3.10. The tested stack is the one listed above. A clean Linux/CUDA
environment can be prepared as follows (Linux commands; the local delivery test was run on Windows):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

The editable install is required because several CLI tests launch `python scripts/...` as a child
process; the child needs `promptguard` to be importable from the environment, not only from the
current working directory. `--no-deps` ensures the pinned runtime packages are not changed.

Model weights are not included in this repository. After setting up the environment:

```bash
cd PromptGuard
python scripts/download_model.py
```

This downloads `THUDM/CogVideoX-2b` at revision
`1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01` into `PromptGuard/model/`. The default path can be
overridden with `--directory`.

## Tests

From `PromptGuard/`:

```bash
python -m pytest -q
```

The delivered checkout was tested on Windows with the local RTX 5070 (compute capability `sm_120`,
which the pinned torch build does not support for kernels). To make the CLI subprocesses resolve the
delivery package, the test invocation used `PYTHONPATH` pointing at `PromptGuard/`:

```powershell
$env:PYTHONPATH = (Resolve-Path .).Path
python -m pytest -q
```

Actual delivery-copy result:

```text
94 passed, 2 skipped in 32.95s
exit code 0
```

The two skips are the CUDA half-precision tests because the local `sm_120` GPU is not supported by
the installed torch build. On the A40 cloud run, the same suite reported **96 passed, 0 failed,
0 skipped** (`stage06` preflight); that result belongs to the remote environment and is not being
re-claimed as a local test result.

## Reproducing a smoke run

After the editable install and model download, a single run can be reproduced with the frozen
settings, for example:

```bash
cd PromptGuard
python scripts/generate.py \
  --model model \
  --prompt "A small red bird flies above a blue car." \
  --mode promptguard \
  --steps 8 --frames 9 --seed 42 \
  --density 0.4 --alpha 0.2 --delta 0.05 \
  --block-size 256 --warmup 1 \
  --guidance 6.0 --vram-gib 23 --offload model \
  --output ../artifacts/smoke/reproduction/readme_prompt.mp4
```

For a dry-run plan of all six selected benchmark records:

```bash
cd PromptGuard
python scripts/plan_benchmark_run.py \
  --manifest ../artifacts/manifests/t2v_compbench_smoke.jsonl \
  --output-dir ../artifacts/dry_runs/reproduction_t2v \
  --dry-run
```

`plan_benchmark_run.py` writes `plan.jsonl` and `run_summary.json`; it does not load a model or
generate video. The smoke run used one independent `generate.py` process per record so that the
PromptGuard `Controller`/`Router` state could not leak between prompts.

## Artifacts and provenance

- `artifacts/environment/smoke_protocol.json` — frozen model revision, prompt IDs, generation
  parameters, output layout, and reporting requirements.
- `artifacts/environment/cloud_run_checklist.md` — the reviewed preflight/run checklist.
- `artifacts/sources/*/SOURCE_RECORD.md` — official repository URLs, pinned revisions, retrieval
  dates, and SHA-256 values.
- `artifacts/dry_runs/` — two small Stage 04 dry-run plans used by
  `tests/test_smoke_protocol.py`. Their local executable/output paths were normalized to
  project-relative values for delivery.

The delivery copies of the generated JSON files only normalize the recorded model path to the
project-relative `model`; benchmark prompts, IDs, measurements, routing data, and generation
settings are unchanged.

## Limitations

- Six benchmark prompts (three from each benchmark) plus two controls; this is not a complete
  benchmark or a statistically meaningful evaluation.
- No official evaluator was integrated or run.
- No training or fine-tuning was performed.
- The reported MP4 sizes and frame statistics belong to the recorded run; generation is stochastic
  and a reproduction with another GPU/driver may differ.
- The A40 test result (96 passed) and the local delivery-copy result (94 passed, 2 skipped) are kept
  distinct; the local skip is a hardware/torch compatibility limitation, not a test failure.

See [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) for the concise experiment summary and
[`artifacts/smoke/stage06_smoke_v1/RUN_REPORT.md`](artifacts/smoke/stage06_smoke_v1/RUN_REPORT.md)
for the per-run table.
