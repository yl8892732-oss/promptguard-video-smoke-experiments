# T2V-CompBench prompt source record

- Official repository: https://github.com/KaiyueSun98/T2V-CompBench
- Default/current branch at retrieval: `V1`
- Resolved revision: `116f996847da14ef1474b8f19ebce5c50da264f3`
  (last commit on `V1`, 2024-08-30T21:48:48Z)
- Retrieved: 2026-10-25 22:48 +08:00 (Asia/Shanghai)
- Retrieval method: direct HTTPS from `raw.githubusercontent.com` at the pinned commit; no
  repository clone, no Git initialization, no GitHub write operation.
- Scope: only the seven `meta_data/*.json` prompt/metadata files below. No videos, model weights,
  evaluator code, or training data were downloaded.

## Downloaded files

| Relative path | Bytes | SHA-256 | Pinned URL |
|---|---:|---|---|
| `meta_data/consistent_attribute_binding.json` | 13,203 | `21f3aec510a72b3bad30e10b7093db2f656111619ea7e821c8aad58519738491` | https://raw.githubusercontent.com/KaiyueSun98/T2V-CompBench/116f996847da14ef1474b8f19ebce5c50da264f3/meta_data/consistent_attribute_binding.json |
| `meta_data/dynamic_attribute_binding.json` | 20,744 | `4b0b91da789723360c06208327894b5e0e32970f2e1e7bffd4f706dba2669822` | https://raw.githubusercontent.com/KaiyueSun98/T2V-CompBench/116f996847da14ef1474b8f19ebce5c50da264f3/meta_data/dynamic_attribute_binding.json |
| `meta_data/spatial_relationships.json` | 16,442 | `5df2e09c46446989c0c326aa1552a846764d1a674f90c5eca60f09b447565778` | https://raw.githubusercontent.com/KaiyueSun98/T2V-CompBench/116f996847da14ef1474b8f19ebce5c50da264f3/meta_data/spatial_relationships.json |
| `meta_data/motion_binding.json` | 18,364 | `0f88822f53499f368578152527b1e69ff2645b669a9fe8036b88a905b16b55be` | https://raw.githubusercontent.com/KaiyueSun98/T2V-CompBench/116f996847da14ef1474b8f19ebce5c50da264f3/meta_data/motion_binding.json |
| `meta_data/action_binding.json` | 28,407 | `84ef02388dbd491b58e9d44d6d1946b11083d3b0f1c84f481bc4c6c68cf765c1` | https://raw.githubusercontent.com/KaiyueSun98/T2V-CompBench/116f996847da14ef1474b8f19ebce5c50da264f3/meta_data/action_binding.json |
| `meta_data/object_interactions.json` | 7,713 | `d07283a1e42346b6b11c48929fb42d15a1bfda4d613ca665001f01da9a20457c` | https://raw.githubusercontent.com/KaiyueSun98/T2V-CompBench/116f996847da14ef1474b8f19ebce5c50da264f3/meta_data/object_interactions.json |
| `meta_data/generative_numeracy.json` | 13,778 | `ade61d9cebd8ced9445a6870feb74daaacaf522cf68f8eb615234f5d47bdc825` | https://raw.githubusercontent.com/KaiyueSun98/T2V-CompBench/116f996847da14ef1474b8f19ebce5c50da264f3/meta_data/generative_numeracy.json |

## Category mapping used by the adapter

| Manifest `group` | Source file |
|---|---|
| `consistent_attribute_binding` | `meta_data/consistent_attribute_binding.json` |
| `dynamic_attribute_binding` | `meta_data/dynamic_attribute_binding.json` |
| `spatial_relationships` | `meta_data/spatial_relationships.json` |
| `motion_binding` | `meta_data/motion_binding.json` |
| `action_binding` | `meta_data/action_binding.json` |
| `object_interactions` | `meta_data/object_interactions.json` |
| `numeracy` | `meta_data/generative_numeracy.json` |

## Count note

At this revision the official README states: “The T2V-CompBench prompt suite includes 700 prompts
covering 7 categories, each with 100 prompts.” Every `meta_data/*.json` file listed above contains
exactly 100 items, and the corresponding `prompts/*.txt` files contain 100 non-empty lines. This
differs from the older workflow text mentioning 1,400 prompts / 200 per category; the adapter uses
the actual pinned source (100 per category, IDs `0001`–`0100`) rather than inventing 100 extra
prompts. The discrepancy is reported in the Stage 03 change log.
