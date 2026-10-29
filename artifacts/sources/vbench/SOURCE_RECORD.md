# VBench prompt source record

- Official repository: https://github.com/Vchitect/VBench
- Branch at retrieval: `master`
- Resolved revision: `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`
  (last commit on `master`, 2026-08-21T15:28:07Z)
- Retrieved: 2026-10-25 22:48 +08:00 (Asia/Shanghai)
- Retrieval method: direct HTTPS from `raw.githubusercontent.com` at the pinned commit; no
  repository clone, no Git initialization, no GitHub write operation.
- Scope: original T2V suite only (`vbench/VBench_full_info.json`). No VBench-Long, VBench-I2V,
  VBench++, VBench-2.0, generated videos, evaluator weights, or dependencies were downloaded.

## Downloaded files

| Relative path | Bytes | SHA-256 | Pinned URL |
|---|---:|---|---|
| `vbench/VBench_full_info.json` | 228,118 | `5dd2de80ee43cda750b2b72ea7023657c0b90d3702041c7e4608c65dbe50dccd` | https://raw.githubusercontent.com/Vchitect/VBench/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench/VBench_full_info.json |

## Structure note

The JSON is a list of 946 items. Every item has `prompt_en` and a `dimension` list; 16 distinct
dimensions occur (`multiple_objects`, `human_action`, `spatial_relationship`, `object_class`,
`color`, `scene`, `temporal_style`, `appearance_style`, `background_consistency`,
`subject_consistency`, `dynamic_degree`, `motion_smoothness`, `temporal_flickering`,
`imaging_quality`, `aesthetic_quality`, `overall_consistency`). Some items also carry
`auxiliary_info`. The adapter preserves the complete original item as `metadata` and the complete
`dimension` list as `group`.
