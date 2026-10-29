"""CPU-only validation of the frozen Stage 05 smoke protocol and cloud checklist."""
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT.parent
PROTOCOL_PATH = WORKSPACE / "artifacts" / "environment" / "smoke_protocol.json"
CHECKLIST_PATH = WORKSPACE / "artifacts" / "environment" / "cloud_run_checklist.md"

EXPECTED_RECORDS = {
    "t2v_compbench": [
        ("consistent_attribute_binding", "0001"),
        ("spatial_relationships", "0001"),
        ("action_binding", "0001"),
    ],
    "vbench": [
        (("multiple_objects",), "0076"),
        (("human_action",), "0158"),
        (("spatial_relationship",), "0863"),
    ],
}
EXPECTED_PROMPTS = {
    ("t2v_compbench", "consistent_attribute_binding", "0001"):
        "A blue car drives past a white picket fence on a sunny day",
    ("t2v_compbench", "spatial_relationships", "0001"): "A dog running on the left of a bicycle",
    ("t2v_compbench", "action_binding", "0001"):
        "A dog runs through a field while a cat climbs a tree",
    ("vbench", ("multiple_objects",), "0076"): "a bird and a cat",
    ("vbench", ("human_action",), "0158"): "A person is riding a bike",
    ("vbench", ("spatial_relationship",), "0863"):
        "a bicycle on the left of a car, front view",
}


def _load_protocol():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _group_key(group):
    return tuple(group) if isinstance(group, list) else group


def test_protocol_required_fields_and_model():
    protocol = _load_protocol()
    assert protocol["schema_version"] == 1
    assert protocol["protocol_id"] == "stage05_smoke_v1"
    assert protocol["created_at"]
    assert protocol["model"] == {
        "id": "THUDM/CogVideoX-2b",
        "revision": "1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01",
        "local_dir": "model",
        "dtype": "float16",
    }
    assert "sm_120" in protocol["runtime"]["incompatible_compute_capabilities"]
    assert "sm_89" in protocol["runtime"]["incompatible_compute_capabilities"]
    assert "sm_80" in protocol["runtime"]["recommended_compute_capabilities"]

    generation = protocol["generation"]
    assert generation["mode"] == "promptguard"
    assert generation["seed"] == 42
    assert generation["resolution"] == [720, 480]
    assert generation["height"] == 480 and generation["width"] == 720
    assert generation["offload"] == "model"
    assert protocol["control"]["prompt"] == "A small red bird flies above a blue car."
    assert protocol["control"]["modes"] == ["dense", "promptguard"]
    assert protocol["run"]["one_process_per_record"] is True
    assert "prompt_id" in protocol["reporting"]["required_stage6_metadata"]
    assert "model_revision" in protocol["reporting"]["required_stage6_metadata"]

    download_script = (PROJECT_ROOT / "scripts" / "download_model.py").read_text(encoding="utf-8")
    assert protocol["model"]["revision"] in download_script
    assert protocol["model"]["id"] in download_script


def test_protocol_selected_records_and_manifest_provenance():
    protocol = _load_protocol()
    for benchmark, expected in EXPECTED_RECORDS.items():
        manifest_info = protocol["manifests"][benchmark]
        manifest = WORKSPACE / manifest_info["path"]
        assert manifest.is_file()
        assert hashlib.sha256(manifest.read_bytes()).hexdigest() == manifest_info["sha256"]
        rows = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == 3

        by_key = {
            (row["benchmark"], _group_key(row["group"]), row["prompt_id"]): row
            for row in rows
        }
        protocol_records = protocol["records"][benchmark]
        assert [
            (_group_key(item["group"]), item["prompt_id"]) for item in protocol_records
        ] == expected
        for group, prompt_id in expected:
            key = (benchmark, group, prompt_id)
            assert key in by_key
            assert by_key[key]["prompt"] == EXPECTED_PROMPTS[key]
            if isinstance(group, tuple):
                assert by_key[key]["group"] == list(group)
            else:
                assert by_key[key]["group"] == group


def test_protocol_generation_constraints_match_dry_run_plans():
    protocol = _load_protocol()
    generation = protocol["generation"]
    assert generation["seed"] == 42
    assert generation["steps"] > generation["warmup"]
    assert generation["frames"] > 0 and (generation["frames"] - 1) % 4 == 0
    assert 0 < generation["vram_gib"] <= 24
    assert 0 < generation["density"] <= 1
    assert 0 <= generation["alpha"] <= 1 and 0 <= generation["delta"] <= 1
    assert generation["block_size"] > 0
    assert generation["warmup"] >= 0
    assert generation["resolution"] == [generation["width"], generation["height"]]

    plan_paths = [
        WORKSPACE / "artifacts" / "dry_runs" / "stage04_t2v_compbench" / "plan.jsonl",
        WORKSPACE / "artifacts" / "dry_runs" / "stage04_vbench" / "plan.jsonl",
    ]
    plan_records = []
    for path in plan_paths:
        assert path.is_file()
        plan_records.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
    assert len(plan_records) == 6
    for plan in plan_records:
        assert plan["seed"] == 42
        for key, value in plan["generation"].items():
            if key == "model":
                assert value == protocol["model"]["local_dir"]
            else:
                assert value == generation[key], key


def test_cloud_checklist_has_required_sections():
    text = CHECKLIST_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    for required in [
        "authorization",
        "24 gib",
        "60–100 gb",
        "nvidia-smi",
        "download_model.py",
        "sm_89",
        "sm_120",
        "one fresh process per record",
        "stage06_smoke_v1",
        "stop the pod",
        "prompt_id",
        "model_revision",
        "sidecar",
    ]:
        assert required.lower() in lowered, required
