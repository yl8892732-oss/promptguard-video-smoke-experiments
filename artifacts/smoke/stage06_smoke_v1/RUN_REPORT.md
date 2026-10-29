# Stage 06 Smoke Run Report

## Scope

This is the per-run report for a **small-sample smoke test** of CogVideoX-2B with PromptGuard. It is
not a complete benchmark or an official evaluation. The run contains two control generations and six
benchmark generations.

## Environment

| Item | Value |
|---|---|
| Model | `THUDM/CogVideoX-2b` |
| Revision | `1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01` |
| GPU | NVIDIA A40, 46,068 MiB |
| GPU capability | 8.6 (`sm_86`) |
| torch | 2.7.1+cu126 |
| diffusers | 0.35.2 |
| transformers | 4.57.0 |
| accelerate | 1.15.0 |
| Python | 3.11.13 |
| Preflight tests | 96 passed, 0 failed, 0 skipped |

## Settings

- mode: `promptguard` for benchmark records; controls used `dense` then `promptguard`
- seed: `42` for all records
- steps: `8`, frames: `9`, resolution: `720x480`
- guidance: `6.0`
- density: `0.4`, alpha: `0.2`, delta: `0.05`
- block size: `256`, warmup: `1`
- offload: `model`, VRAM cap: `23 GiB`

Each row below is a separate `generate.py` process.

## Results

| Run | Benchmark | Group | Prompt ID | MP4 bytes | JSON bytes | finite | pixel std | decoded frames | peak reserved GiB | routing layers |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|
| control dense | control | dense | `readme_prompt` | 8,500 | 3,273 | true | 0.089036 | 9 | 10.83 | 0 |
| control promptguard | control | promptguard | `readme_prompt` | 27,662 | 78,458 | true | 0.104412 | 9 | 10.83 | 30 |
| T2V consistent_attribute_binding | t2v_compbench | consistent_attribute_binding | `0001` | 207,253 | 78,728 | true | 0.357982 | 9 | 10.83 | 30 |
| T2V spatial_relationships | t2v_compbench | spatial_relationships | `0001` | 291,798 | 78,505 | true | 0.292934 | 9 | 10.83 | 30 |
| T2V action_binding | t2v_compbench | action_binding | `0001` | 259,015 | 78,615 | true | 0.308758 | 9 | 10.83 | 30 |
| VBench multiple_objects | vbench | multiple_objects | `0076` | 110,964 | 79,515 | true | 0.263344 | 9 | 10.83 | 30 |
| VBench human_action | vbench | human_action | `0158` | 128,120 | 78,434 | true | 0.281149 | 9 | 10.83 | 30 |
| VBench spatial_relationship | vbench | spatial_relationship | `0863` | 137,437 | 78,533 | true | 0.292944 | 9 | 10.83 | 30 |

All eight processes exited 0. All decoded MP4s are 720×480, 8 fps, 9 frames, with non-constant
content. All JSONs have `finite: true`, `video_shape: [9, 480, 720, 3]`, and peak reserved VRAM
below the 23 GiB cap. PromptGuard runs have non-empty routing reports; the dense control has none.

## Output layout

```text
artifacts/smoke/stage06_smoke_v1/
├── control/dense/readme_prompt.mp4|.json
├── control/promptguard/readme_prompt.mp4|.json
├── t2v_compbench/consistent_attribute_binding/0001.mp4|.json
├── t2v_compbench/spatial_relationships/0001.mp4|.json
├── t2v_compbench/action_binding/0001.mp4|.json
├── vbench/multiple_objects/0076.mp4|.json
├── vbench/human_action/0158.mp4|.json
└── vbench/spatial_relationship/0863.mp4|.json
```

Each JSON retains the prompt, model/mode/config/seed, frames/steps/guidance, GPU, VRAM peaks,
video shape, `finite`, pixel statistics, per-step metrics, semantic units, and routing report. The
delivery copy additionally records the benchmark/group/prompt ID, model revision, protocol ID, run
timestamp, and a code-version hash string.

## Validation

- Paired MP4/JSON files exist for every row and are non-empty.
- JSON values for finite, shape, seed, settings, and routing were checked.
- MP4s were decoded and checked for expected dimensions, frame count, and non-constant content.
- The selected benchmark IDs and prompts match `artifacts/manifests/`.

## Limitations

- Six benchmark prompts plus two controls; not a full benchmark.
- No official evaluator or metric run.
- No training or fine-tuning.
- Repeated generation may differ because sampling is stochastic.
- Generation was performed on an A40 cloud GPU; the local delivery-copy test skips two CUDA tests
  because the local `sm_120` GPU is unsupported by the pinned torch build.
