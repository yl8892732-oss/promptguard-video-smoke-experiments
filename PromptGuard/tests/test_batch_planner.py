import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from promptguard import benchmarks as b

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLANNER_PATH = PROJECT_ROOT / "scripts" / "plan_benchmark_run.py"


def _load_planner():
    spec = importlib.util.spec_from_file_location("plan_benchmark_run", PLANNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


planner = _load_planner()


def _row(
    *,
    benchmark=b.T2V_COMPBENCH,
    group="action_binding",
    prompt_id="0001",
    prompt="a small red bird flies above a blue car",
    index=0,
):
    return {
        "schema_version": 1,
        "benchmark": benchmark,
        "group": group,
        "prompt_id": prompt_id,
        "prompt": prompt,
        "source": {
            "path": "meta_data/synthetic.json",
            "index": index,
            "sha256": "0" * 64,
        },
        "metadata": {"tag": f"{benchmark}-{prompt_id}"},
    }


def _write_manifest(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _args(manifest, output, *extra):
    args = [
        "--manifest", str(manifest),
        "--output-dir", str(output),
        "--dry-run",
    ]
    args.extend(str(item) for item in extra)
    return args


def _read_plan(output):
    return [json.loads(line) for line in (output / "plan.jsonl").read_text(encoding="utf-8").splitlines()]


def _assert_no_outputs(output):
    assert not (output / "plan.jsonl").exists()
    assert not (output / "run_summary.json").exists()


def _flag(argv, name):
    return argv[argv.index(name) + 1]


def test_valid_plan_preserves_provenance_semantics_and_argv(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        _row(group="action_binding", prompt_id="0001"),
        _row(group="spatial_relationships", prompt_id="0002", prompt="A dog running on the left of a bicycle"),
    ]
    _write_manifest(manifest, rows)
    output = tmp_path / "run"
    result = planner.main(
        _args(
            manifest,
            output,
            "--model", "model-x",
            "--mode", "promptguard",
            "--steps", "4",
            "--frames", "9",
            "--seed", "42",
            "--density", "0.4",
            "--alpha", "0.2",
            "--delta", "0.05",
            "--block-size", "256",
            "--warmup", "1",
            "--guidance", "6",
            "--vram-gib", "23",
            "--offload", "sequential",
        )
    )
    assert result == 0
    plans = _read_plan(output)
    assert len(plans) == 2
    assert [plan["prompt_id"] for plan in plans] == ["0001", "0002"]
    assert [plan["group"] for plan in plans] == ["action_binding", "spatial_relationships"]
    assert plans[0]["source"] == rows[0]["source"]
    assert plans[0]["metadata"] == rows[0]["metadata"]
    assert plans[0]["semantic_units"] == [
        {"label": "small red bird", "spans": [[2, 16]]},
        {"label": "blue car", "spans": [[31, 39]]},
        {"label": "bird flies", "spans": [[12, 16], [17, 22]]},
        {"label": "bird above car", "spans": [[12, 16], [23, 28], [36, 39]]},
    ]
    assert all(plan["semantic_units"] for plan in plans)
    assert plans[0]["seed"] == 42
    assert plans[0]["generation"] == {
        "model": "model-x",
        "mode": "promptguard",
        "steps": 4,
        "frames": 9,
        "seed": 42,
        "density": 0.4,
        "alpha": 0.2,
        "delta": 0.05,
        "block_size": 256,
        "warmup": 1,
        "guidance": 6.0,
        "vram_gib": 23.0,
        "offload": "sequential",
    }
    assert plans[0]["output_path"] == "t2v_compbench/action_binding/0001.mp4"
    assert plans[1]["output_path"] == "t2v_compbench/spatial_relationships/0002.mp4"
    assert plans[0]["argv"][:2] == [sys.executable, "scripts/generate.py"]
    assert _flag(plans[0]["argv"], "--prompt") == rows[0]["prompt"]
    assert _flag(plans[0]["argv"], "--mode") == "promptguard"
    assert _flag(plans[0]["argv"], "--steps") == "4"
    assert _flag(plans[0]["argv"], "--frames") == "9"
    assert _flag(plans[0]["argv"], "--seed") == "42"
    assert _flag(plans[0]["argv"], "--model") == "model-x"
    assert _flag(plans[0]["argv"], "--block-size") == "256"
    assert _flag(plans[0]["argv"], "--offload") == "sequential"
    assert _flag(plans[0]["argv"], "--output") == str(output / "t2v_compbench/action_binding/0001.mp4")

    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["record_count"] == 2
    assert summary["mode"] == "promptguard"
    assert summary["seed"] == 42
    assert summary["dry_run"] is True
    assert summary["model_operations"] == 0
    assert summary["subprocess_operations"] == 0
    assert summary["cuda_operations"] == 0
    assert summary["video_operations"] == 0


def test_vbench_group_is_preserved_but_path_component_is_joined(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    row = _row(
        benchmark=b.VBENCH,
        group=["human_action", "dynamic_degree"],
        prompt_id="0158",
        prompt="A person is riding a bike",
    )
    _write_manifest(manifest, [row])
    output = tmp_path / "run"
    assert planner.main(_args(manifest, output)) == 0
    plan = _read_plan(output)[0]
    assert plan["group"] == ["human_action", "dynamic_degree"]
    assert plan["metadata"] == row["metadata"]
    assert plan["output_path"] == "vbench/human_action__dynamic_degree/0158.mp4"


def test_malformed_json_reports_line_and_creates_no_plan(tmp_path, capsys):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{not json\n", encoding="utf-8")
    output = tmp_path / "run"
    assert planner.main(_args(manifest, output)) == 1
    error = capsys.readouterr().err
    assert ":1:" in error and "malformed JSON" in error
    _assert_no_outputs(output)


def test_missing_schema_field_reports_line_and_prompt_id(tmp_path, capsys):
    manifest = tmp_path / "manifest.jsonl"
    row = _row()
    row.pop("source")
    _write_manifest(manifest, [row])
    output = tmp_path / "run"
    assert planner.main(_args(manifest, output)) == 1
    error = capsys.readouterr().err
    assert ":1," in error and "prompt_id '0001'" in error and "missing" in error
    _assert_no_outputs(output)


def test_blank_prompt_and_semantic_failure_abort(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [_row(prompt="   ")])
    output = tmp_path / "run"
    assert planner.main(_args(manifest, output)) == 1
    _assert_no_outputs(output)

    _write_manifest(manifest, [_row(prompt="12345")])
    output = tmp_path / "run2"
    assert planner.main(_args(manifest, output)) == 1
    _assert_no_outputs(output)


@pytest.mark.parametrize(
    "row",
    [
        _row(group="../evil"),
        _row(benchmark=b.VBENCH, group=["../evil"], prompt="a bird and a cat"),
    ],
)
def test_unsafe_group_is_rejected(tmp_path, row):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [row])
    output = tmp_path / "run"
    assert planner.main(_args(manifest, output)) == 1
    _assert_no_outputs(output)


def test_duplicate_logical_record_is_rejected(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    row = _row()
    _write_manifest(manifest, [row, dict(row)])
    output = tmp_path / "run"
    assert planner.main(_args(manifest, output)) == 1
    _assert_no_outputs(output)


@pytest.mark.parametrize(
    "extra",
    [
        ["--frames", "10"],
        ["--vram-gib", "0"],
        ["--steps", "1", "--warmup", "1"],
        ["--density", "1.5"],
        ["--alpha", "-0.1"],
    ],
)
def test_invalid_generation_options_abort(tmp_path, extra):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [_row()])
    output = tmp_path / "run"
    assert planner.main(_args(manifest, output, *extra)) == 1
    _assert_no_outputs(output)


def test_dry_run_is_mandatory(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [_row()])
    output = tmp_path / "run"
    assert planner.main(["--manifest", str(manifest), "--output-dir", str(output)]) == 1
    _assert_no_outputs(output)


def test_overwrite_protection_and_deterministic_repeat(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [_row()])
    output = tmp_path / "run"
    args = _args(manifest, output)
    assert planner.main(args) == 0
    plan_before = (output / "plan.jsonl").read_bytes()
    summary_before = (output / "run_summary.json").read_bytes()

    assert planner.main(args) == 1
    assert (output / "plan.jsonl").read_bytes() == plan_before
    assert (output / "run_summary.json").read_bytes() == summary_before

    assert planner.main([*args, "--overwrite"]) == 0
    assert (output / "plan.jsonl").read_bytes() == plan_before
    assert (output / "run_summary.json").read_bytes() == summary_before


def test_planner_never_starts_a_subprocess(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [_row()])
    output = tmp_path / "run"

    def fail(*args, **kwargs):
        raise AssertionError("the planner must not start a subprocess")

    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(subprocess, "Popen", fail)
    assert planner.main(_args(manifest, output)) == 0


def test_fresh_import_does_not_load_model_packages():
    code = (
        "import importlib.util, sys; "
        f"spec = importlib.util.spec_from_file_location('planner', r'{PLANNER_PATH}'); "
        "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "bad = [name for name in ('torch', 'diffusers', 'transformers') if name in sys.modules]; "
        "assert not bad, bad"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "filename, expected_ids, expected_groups",
    [
        (
            "t2v_compbench_smoke.jsonl",
            ["0001", "0001", "0001"],
            ["consistent_attribute_binding", "spatial_relationships", "action_binding"],
        ),
        (
            "vbench_smoke.jsonl",
            ["0076", "0158", "0863"],
            [["multiple_objects"], ["human_action"], ["spatial_relationship"]],
        ),
    ],
)
def test_real_smoke_manifests_plan_three_records(
    tmp_path, filename, expected_ids, expected_groups
):
    manifest = PROJECT_ROOT.parent / "artifacts" / "manifests" / filename
    assert manifest.is_file(), manifest
    output = tmp_path / "run"
    assert planner.main(_args(manifest, output)) == 0
    plans = _read_plan(output)
    assert len(plans) == 3
    assert [plan["prompt_id"] for plan in plans] == expected_ids
    assert [plan["group"] for plan in plans] == expected_groups
    assert all(plan["seed"] == 42 for plan in plans)
    assert all(plan["semantic_units"] for plan in plans)
    assert all(plan["output_path"].endswith(".mp4") for plan in plans)
    assert all(not Path(plan["output_path"]).is_absolute() for plan in plans)
    assert all(".." not in Path(plan["output_path"]).parts for plan in plans)


def test_cli_help_and_missing_dry_run_are_non_generating(tmp_path):
    help_result = subprocess.run(
        [sys.executable, str(PLANNER_PATH), "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    assert help_result.returncode == 0
    assert "--dry-run" in help_result.stdout

    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [_row()])
    output = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            str(PLANNER_PATH),
            "--manifest", str(manifest),
            "--output-dir", str(output),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "--dry-run is required" in result.stderr
    _assert_no_outputs(output)
