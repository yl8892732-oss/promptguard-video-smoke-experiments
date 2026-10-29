# Cloud Run Checklist (Delivery Copy)

This is a condensed, credential-free checklist for reproducing the six-record smoke run. It is not a
cloud operation log and contains no Pod, SSH, token, or billing details.

## Authorization and hardware

- [ ] The user has provided a GPU environment and authorization for the run.
- [ ] Use a single GPU with at least 24 GiB VRAM; 60–100 GB persistent storage is recommended.
- [ ] Use a compute capability supported by the installed torch build. The tested stack supports
      `sm_80`, `sm_86`, and `sm_90`; it does not support `sm_89` (RTX 4090/L40S) or `sm_120`
      (RTX 50-series) because of the pinned torch wheel.
- [ ] Expose only the required access port and keep any credentials outside the repository.

## Preflight

```bash
cd PromptGuard
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name()); print(torch.cuda.get_device_capability())"
python -m pytest -q
```

Stop if CUDA is unavailable, the compute capability is unsupported, or a non-CUDA test fails. Record
the GPU/driver/CUDA/Python/torch/diffusers/transformers versions.

## Model

```bash
cd PromptGuard
python scripts/download_model.py
```

This downloads `THUDM/CogVideoX-2b` at the pinned revision into `model/`; model weights are not
included in this repository. Do not download datasets or evaluator weights.

## Generation

Run from `PromptGuard/`. Use one fresh process per record; do not batch prompts through one process.
Use the frozen settings from `artifacts/environment/smoke_protocol.json` and write outputs under
`artifacts/smoke/stage06_smoke_v1/`:

- `control/dense/readme_prompt.mp4|.json`
- `control/promptguard/readme_prompt.mp4|.json`
- `t2v_compbench/consistent_attribute_binding/0001.mp4|.json`
- `t2v_compbench/spatial_relationships/0001.mp4|.json`
- `t2v_compbench/action_binding/0001.mp4|.json`
- `vbench/multiple_objects/0076.mp4|.json`
- `vbench/human_action/0158.mp4|.json`
- `vbench/spatial_relationship/0863.mp4|.json`

Run the control prompt first in dense mode, then in promptguard mode. Run the T2V-CompBench and
VBench records in `promptguard` mode; stop on the first failure and preserve the traceback.

## Per-record sidecar

`scripts/generate.py` writes a same-stem `.json` with the numeric/model results. Merge or record the
`benchmark`, `group`, `prompt_id`, model revision, timestamp, and code version from the protocol.
Do not use a prompt_id, model_revision, or sidecar value that contradicts the manifest.

## Verification and stop

- [ ] Every MP4 is non-empty and decodes to 720×480, 9 frames, non-constant content.
- [ ] Every JSON has `finite: true`, `video_shape: [9, 480, 720, 3]`, and a valid routing report.
- [ ] The 6 benchmark records and 2 controls have matching MP4/JSON stems.
- [ ] Synchronize artifacts, verify them, and stop the Pod after the run to control cost.
- [ ] Do not expose or commit credentials, tokens, SSH keys, or billing information.
