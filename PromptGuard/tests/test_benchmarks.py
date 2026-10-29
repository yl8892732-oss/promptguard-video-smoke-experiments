import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from promptguard import benchmarks as b

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _record(prompt_id="0001", group="numeracy") -> b.ManifestRecord:
    return b.ManifestRecord(
        schema_version=1,
        benchmark=b.T2V_COMPBENCH,
        group=group,
        prompt_id=prompt_id,
        prompt="a prompt",
        source={"path": "meta_data/generative_numeracy.json", "index": 0, "sha256": "0" * 64},
        metadata={},
    )


def _make_t2v_fixture(root: Path, groups, per_group=100) -> Path:
    source = root / "t2v"
    for group in groups:
        items = [
            {"prompt": f"{group} prompt {index}", "tag": f"{group}-{index}"}
            for index in range(1, per_group + 1)
        ]
        _write_json(source / b.T2V_CATEGORY_SOURCES[group], items)
    return source


def _make_vbench_fixture(root: Path) -> Path:
    source = root / "vbench"
    items = [
        {
            "prompt_en": "a bird and a cat",
            "dimension": ["multiple_objects"],
            "auxiliary_info": {"multiple_objects": {"object": "bird and cat"}},
        },
        {
            "prompt_en": "A person is riding a bike",
            "dimension": ["human_action", "dynamic_degree"],
        },
        {
            "prompt_en": "a bicycle on the left of a car, front view",
            "dimension": ["spatial_relationship"],
            "auxiliary_info": {
                "spatial_relationship": {
                    "spatial_relationship": {
                        "object_a": "bicycle",
                        "object_b": "car",
                        "relationship": "on the left of",
                    }
                }
            },
        },
    ]
    _write_json(source / b.VBENCH_INFO_PATH, items)
    return source


def _run_cli(*args) -> subprocess.CompletedProcess:
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "build_benchmark_manifest.py")]
    command.extend(str(arg) for arg in args)
    return subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)


def test_t2v_category_mapping_order_ids_and_metadata(tmp_path):
    fixtures = {}
    for group in b.T2V_CATEGORIES:
        items = [
            {"prompt": f"{group} prompt {index}", "phrases": f"phrase {index}"}
            for index in (1, 2)
        ]
        digest = _write_json(tmp_path / b.T2V_CATEGORY_SOURCES[group], items)
        fixtures[group] = (items, digest)

    records = b.load_t2v_compbench(tmp_path, expected_per_category=2, project_root=tmp_path)

    assert len(records) == 2 * len(b.T2V_CATEGORIES)
    assert [record.group for record in records] == [
        group for group in b.T2V_CATEGORIES for _ in range(2)
    ]
    for group in b.T2V_CATEGORIES:
        subset = [record for record in records if record.group == group]
        items, digest = fixtures[group]
        assert [record.prompt_id for record in subset] == ["0001", "0002"]
        assert [record.prompt for record in subset] == [item["prompt"] for item in items]
        assert [record.metadata for record in subset] == items
        assert subset[0].source == {
            "path": b.T2V_CATEGORY_SOURCES[group],
            "index": 0,
            "sha256": digest,
        }
        assert subset[1].source["index"] == 1


def test_t2v_group_selection_is_official_order_and_numeracy_maps_to_generative_file(tmp_path):
    _write_json(
        tmp_path / "meta_data" / "generative_numeracy.json",
        [{"prompt": "two cats", "numbers": "2", "objects": "cat"}],
    )
    _write_json(
        tmp_path / "meta_data" / "consistent_attribute_binding.json",
        [{"prompt": "a red cube", "phrases": "a red cube"}],
    )
    records = b.load_t2v_compbench(
        tmp_path,
        groups="numeracy,consistent_attribute_binding",
        expected_per_category=1,
        project_root=tmp_path,
    )
    assert [record.group for record in records] == [
        "consistent_attribute_binding",
        "numeracy",
    ]
    assert records[1].metadata == {"prompt": "two cats", "numbers": "2", "objects": "cat"}
    assert records[1].source["path"] == "meta_data/generative_numeracy.json"


def test_vbench_prompt_dimensions_and_auxiliary_info_are_preserved(tmp_path):
    source = _make_vbench_fixture(tmp_path)
    records = b.load_vbench(source, project_root=tmp_path)

    assert [record.prompt_id for record in records] == ["0001", "0002", "0003"]
    assert [record.prompt for record in records] == [
        "a bird and a cat",
        "A person is riding a bike",
        "a bicycle on the left of a car, front view",
    ]
    assert records[1].group == ("human_action", "dynamic_degree")
    assert records[0].group == ("multiple_objects",)
    assert records[0].metadata["auxiliary_info"] == {
        "multiple_objects": {"object": "bird and cat"}
    }
    assert records[2].metadata["auxiliary_info"]["spatial_relationship"][
        "spatial_relationship"
    ]["relationship"] == "on the left of"
    assert records[0].source["path"].endswith(b.VBENCH_INFO_PATH)
    assert records[0].source["index"] == 0


def test_vbench_dimension_membership_filter(tmp_path):
    source = _make_vbench_fixture(tmp_path)
    selected = b.load_vbench(
        source, dimensions="multiple_objects,human_action", project_root=tmp_path
    )
    assert [record.prompt_id for record in selected] == ["0001", "0002"]

    with pytest.raises(b.BenchmarkSourceError, match="unknown VBench dimension"):
        b.load_vbench(source, dimensions="not_a_dimension", project_root=tmp_path)


@pytest.mark.parametrize(
    "item, message",
    [
        ({}, "missing or blank prompt"),
        ({"prompt": "   "}, "missing or blank prompt"),
        ({"prompt": 123}, "missing or blank prompt"),
        ("not an object", "unexpected item structure"),
    ],
)
def test_t2v_rejects_bad_items(tmp_path, item, message):
    _write_json(tmp_path / "meta_data" / "generative_numeracy.json", [item])
    with pytest.raises(b.BenchmarkSourceError, match=message):
        b.load_t2v_compbench(
            tmp_path, groups=["numeracy"], expected_per_category=1, project_root=tmp_path
        )


@pytest.mark.parametrize(
    "item, message",
    [
        ({}, "missing or blank prompt"),
        ({"prompt_en": "  ", "dimension": ["multiple_objects"]}, "missing or blank prompt"),
        ({"prompt_en": "x", "dimension": "multiple_objects"}, "invalid dimension list"),
        ({"prompt_en": "x", "dimension": []}, "invalid dimension list"),
        ({"prompt_en": "x", "dimension": [""]}, "invalid dimension list"),
        ("not an object", "unexpected item structure"),
    ],
)
def test_vbench_rejects_bad_items(tmp_path, item, message):
    _write_json(tmp_path / b.VBENCH_INFO_PATH, [item])
    with pytest.raises(b.BenchmarkSourceError, match=message):
        b.load_vbench(tmp_path, project_root=tmp_path)


def test_t2v_malformed_json_and_wrong_structure(tmp_path):
    path = tmp_path / "meta_data" / "generative_numeracy.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(b.BenchmarkSourceError, match="malformed JSON.*numeracy"):
        b.load_t2v_compbench(tmp_path, groups=["numeracy"], project_root=tmp_path)

    _write_json(path, {"prompt": "not a list"})
    with pytest.raises(b.BenchmarkSourceError, match="expected a JSON list"):
        b.load_t2v_compbench(tmp_path, groups=["numeracy"], project_root=tmp_path)


def test_t2v_missing_file_and_unexpected_count(tmp_path):
    with pytest.raises(b.BenchmarkSourceError, match="missing source file.*numeracy"):
        b.load_t2v_compbench(tmp_path, groups=["numeracy"], project_root=tmp_path)

    _write_json(
        tmp_path / "meta_data" / "generative_numeracy.json",
        [{"prompt": "one"}, {"prompt": "two"}],
    )
    with pytest.raises(b.BenchmarkSourceError, match="unexpected item count 2.*numeracy"):
        b.load_t2v_compbench(
            tmp_path, groups=["numeracy"], expected_per_category=1, project_root=tmp_path
        )


def test_unknown_group_and_empty_filter_fail(tmp_path):
    with pytest.raises(b.BenchmarkSourceError, match="unknown T2V-CompBench group"):
        b.load_t2v_compbench(tmp_path, groups=["not_a_category"])
    with pytest.raises(b.BenchmarkSourceError, match="empty name"):
        b.load_t2v_compbench(tmp_path, groups="numeracy,")


def test_duplicate_ids_are_rejected():
    with pytest.raises(b.BenchmarkSourceError, match="duplicate prompt_id '0001'"):
        b.validate_records([_record(), _record()])


def test_invalid_filter_inputs_are_rejected():
    records = [_record("0001"), _record("0002")]
    with pytest.raises(b.BenchmarkSourceError, match="unknown prompt ID"):
        b.apply_filters(records, ids="9999")
    with pytest.raises(b.BenchmarkSourceError, match="empty name"):
        b.apply_filters(records, ids="0001,")
    with pytest.raises(b.BenchmarkSourceError, match="limit must be at least 1"):
        b.apply_filters(records, limit=0)
    with pytest.raises(b.BenchmarkSourceError, match="per_group_limit must be at least 1"):
        b.apply_filters(records, per_group_limit=0)


def test_manifest_record_validates_source_and_prompt():
    with pytest.raises(ValueError, match="absolute Windows path"):
        b.ManifestRecord(
            1,
            b.T2V_COMPBENCH,
            "numeracy",
            "0001",
            "prompt",
            {"path": "C:/x.json", "index": 0, "sha256": "0" * 64},
            {},
        )
    with pytest.raises(ValueError, match="zero-padded"):
        b.ManifestRecord(
            1,
            b.T2V_COMPBENCH,
            "numeracy",
            "1",
            "prompt",
            {"path": "x.json", "index": 0, "sha256": "0" * 64},
            {},
        )


def test_cli_limit_ids_order_and_schema(tmp_path):
    groups = [
        "consistent_attribute_binding",
        "action_binding",
        "spatial_relationships",
    ]
    source = _make_t2v_fixture(tmp_path, groups)
    output = tmp_path / "manifest.jsonl"

    result = _run_cli(
        "--benchmark", b.T2V_COMPBENCH,
        "--source-root", source,
        "--output", output,
        "--groups", ",".join(groups),
        "--limit", "2",
    )
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["group"] for row in rows] == [
        "consistent_attribute_binding",
        "consistent_attribute_binding",
    ]
    assert [row["prompt_id"] for row in rows] == ["0001", "0002"]
    assert set(rows[0]) == {
        "schema_version", "benchmark", "group", "prompt_id", "prompt", "source", "metadata"
    }
    assert rows[0]["schema_version"] == 1
    assert rows[0]["benchmark"] == b.T2V_COMPBENCH
    assert set(rows[0]["source"]) == {"path", "index", "sha256"}
    assert rows[0]["source"]["path"].endswith("meta_data/consistent_attribute_binding.json")
    assert len(rows[0]["source"]["sha256"]) == 64
    assert rows[0]["metadata"]["tag"] == "consistent_attribute_binding-1"

    result = _run_cli(
        "--benchmark", b.T2V_COMPBENCH,
        "--source-root", source,
        "--output", output,
        "--groups", "consistent_attribute_binding",
        "--ids", "0002",
        "--overwrite",
    )
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["prompt_id"] == "0002"


def test_cli_per_group_limit_and_deterministic_output(tmp_path):
    groups = [
        "consistent_attribute_binding",
        "action_binding",
        "spatial_relationships",
    ]
    source = _make_t2v_fixture(tmp_path, groups)
    first, second = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    common = [
        "--benchmark", b.T2V_COMPBENCH,
        "--source-root", source,
        "--groups", ",".join(groups),
        "--per-group-limit", "1",
    ]
    assert _run_cli(*common, "--output", first).returncode == 0
    assert _run_cli(*common, "--output", second).returncode == 0
    assert first.read_bytes() == second.read_bytes()
    rows = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
    assert [row["prompt_id"] for row in rows] == ["0001", "0001", "0001"]
    assert [row["group"] for row in rows] == [
        "consistent_attribute_binding",
        "spatial_relationships",
        "action_binding",
    ]


def test_cli_overwrite_protection(tmp_path):
    source = _make_t2v_fixture(tmp_path, ["consistent_attribute_binding"], per_group=100)
    output = tmp_path / "manifest.jsonl"
    args = [
        "--benchmark", b.T2V_COMPBENCH,
        "--source-root", source,
        "--output", output,
        "--groups", "consistent_attribute_binding",
    ]
    assert _run_cli(*args).returncode == 0
    before = output.read_bytes()
    result = _run_cli(*args)
    assert result.returncode == 1
    assert "refusing to overwrite" in result.stderr
    assert output.read_bytes() == before


def test_cli_rejects_benchmark_inappropriate_filters(tmp_path):
    t2v = _make_t2v_fixture(tmp_path, ["consistent_attribute_binding"], per_group=100)
    vbench = _make_vbench_fixture(tmp_path)
    output = tmp_path / "out.jsonl"
    result = _run_cli(
        "--benchmark", b.T2V_COMPBENCH,
        "--source-root", t2v,
        "--output", output,
        "--dimensions", "multiple_objects",
    )
    assert result.returncode == 1 and "--dimensions is VBench-only" in result.stderr
    result = _run_cli(
        "--benchmark", b.VBENCH,
        "--source-root", vbench,
        "--output", output,
        "--groups", "consistent_attribute_binding",
    )
    assert result.returncode == 1 and "--groups is T2V-CompBench-only" in result.stderr


def test_cli_vbench_per_group_limit(tmp_path):
    source = _make_vbench_fixture(tmp_path)
    output = tmp_path / "vbench.jsonl"
    result = _run_cli(
        "--benchmark", b.VBENCH,
        "--source-root", source,
        "--output", output,
        "--per-group-limit", "1",
    )
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["prompt_id"] for row in rows] == ["0001", "0002", "0003"]


def test_cli_vbench_dimension_and_ids(tmp_path):
    source = _make_vbench_fixture(tmp_path)
    output = tmp_path / "vbench.jsonl"
    result = _run_cli(
        "--benchmark", b.VBENCH,
        "--source-root", source,
        "--output", output,
        "--dimensions", "multiple_objects,human_action,spatial_relationship",
        "--ids", "0001,0002,0003",
    )
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["prompt_id"] for row in rows] == ["0001", "0002", "0003"]
    assert rows[1]["group"] == ["human_action", "dynamic_degree"]
    assert rows[0]["metadata"]["auxiliary_info"]["multiple_objects"]["object"] == "bird and cat"


def test_cli_unknown_id_and_missing_source_fail(tmp_path):
    source = _make_t2v_fixture(tmp_path, ["consistent_attribute_binding"], per_group=100)
    output = tmp_path / "out.jsonl"
    result = _run_cli(
        "--benchmark", b.T2V_COMPBENCH,
        "--source-root", source,
        "--output", output,
        "--groups", "consistent_attribute_binding",
        "--ids", "9999",
    )
    assert result.returncode == 1 and "unknown prompt ID" in result.stderr
    result = _run_cli(
        "--benchmark", b.T2V_COMPBENCH,
        "--source-root", tmp_path / "does-not-exist",
        "--output", output,
    )
    assert result.returncode == 1 and "missing source file" in result.stderr


def test_adapter_import_does_not_require_torch():
    code = (
        "import sys; import promptguard.benchmarks; "
        "bad = [name for name in ('torch', 'diffusers', 'transformers') if name in sys.modules]; "
        "assert not bad, bad"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert _run_cli("--help").returncode == 0
