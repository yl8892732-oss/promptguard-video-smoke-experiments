"""Small, CPU-only adapters for T2V-CompBench and VBench prompt metadata.

This module deliberately depends on the Python standard library only. Importing
it must not import torch, diffusers, transformers, CUDA, or generation code.
Official prompt text, IDs, groups, and metadata are preserved exactly; the
adapters only add a stable manifest envelope around each source item.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SCHEMA_VERSION = 1
T2V_COMPBENCH = "t2v_compbench"
VBENCH = "vbench"

# Explicit mapping from the seven documented benchmark categories to the
# official JSON files.  The official V1 tree names the numeracy file
# "generative_numeracy.json", while the category itself is "numeracy".
T2V_CATEGORY_SOURCES: dict[str, str] = {
    "consistent_attribute_binding": "meta_data/consistent_attribute_binding.json",
    "dynamic_attribute_binding": "meta_data/dynamic_attribute_binding.json",
    "spatial_relationships": "meta_data/spatial_relationships.json",
    "motion_binding": "meta_data/motion_binding.json",
    "action_binding": "meta_data/action_binding.json",
    "object_interactions": "meta_data/object_interactions.json",
    "numeracy": "meta_data/generative_numeracy.json",
}
T2V_CATEGORIES: tuple[str, ...] = tuple(T2V_CATEGORY_SOURCES)
VBENCH_INFO_PATH = "vbench/VBench_full_info.json"

# The pinned official T2V-CompBench revision has 100 prompts per category
# (700 total), despite the older workflow text mentioning 200 per category.
T2V_EXPECTED_PER_CATEGORY = 100

_ID_RE = re.compile(r"\d{4,}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class BenchmarkSourceError(ValueError):
    """Raised when an official source or a manifest filter is invalid."""


@dataclass(frozen=True)
class ManifestRecord:
    """One row of the unified JSONL manifest."""

    schema_version: int
    benchmark: str
    group: str | tuple[str, ...]
    prompt_id: str
    prompt: str
    source: dict
    metadata: dict

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version!r}")
        if self.benchmark not in (T2V_COMPBENCH, VBENCH):
            raise ValueError(f"unsupported benchmark {self.benchmark!r}")
        if self.benchmark == T2V_COMPBENCH:
            if not isinstance(self.group, str) or not self.group.strip():
                raise ValueError("T2V-CompBench group must be a non-empty string")
        elif not isinstance(self.group, tuple) or not self.group:
            raise ValueError("VBench group must be a non-empty dimension tuple")
        if self.benchmark == VBENCH and any(
            not isinstance(dimension, str) or not dimension.strip()
            for dimension in self.group
        ):
            raise ValueError("VBench group contains an invalid dimension name")
        if not isinstance(self.prompt_id, str) or not _ID_RE.fullmatch(self.prompt_id):
            raise ValueError(f"prompt_id must be zero-padded digits, got {self.prompt_id!r}")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        _validate_source(self.source)
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a JSON object")
        try:
            json.dumps(self.metadata, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"metadata is not JSON serializable: {exc}") from exc
        # Keep the record independent from caller-owned mutable objects.
        object.__setattr__(self, "source", dict(self.source))
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))

    def to_dict(self) -> dict:
        group = list(self.group) if isinstance(self.group, tuple) else self.group
        return {
            "schema_version": self.schema_version,
            "benchmark": self.benchmark,
            "group": group,
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "source": dict(self.source),
            "metadata": copy.deepcopy(self.metadata),
        }


def _validate_source(source) -> None:
    if not isinstance(source, dict):
        raise ValueError("source must be a JSON object")
    keys = set(source)
    if keys != {"path", "index", "sha256"}:
        raise ValueError(f"source must have exactly path/index/sha256, got {sorted(keys)}")
    if not isinstance(source["path"], str) or not source["path"] or source["path"].startswith(("/", "\\")):
        raise ValueError("source.path must be a non-empty relative path")
    if re.match(r"^[A-Za-z]:", source["path"]):
        raise ValueError("source.path must not be an absolute Windows path")
    if isinstance(source["index"], bool) or not isinstance(source["index"], int) or source["index"] < 0:
        raise ValueError("source.index must be a non-negative integer")
    if not isinstance(source["sha256"], str) or not _SHA256_RE.fullmatch(source["sha256"]):
        raise ValueError("source.sha256 must be 64 lowercase hexadecimal characters")


def _default_project_root() -> Path:
    # promptguard/benchmarks.py -> video_PromptGuard (the workspace root).
    return Path(__file__).resolve().parents[2]


def _source_path(path: Path, source_root: Path, project_root: Path | None) -> str:
    """Return a portable source path, preferring project-relative form."""
    resolved = path.resolve()
    base = Path(project_root).resolve() if project_root is not None else _default_project_root()
    for candidate in (base, source_root.resolve().parent):
        try:
            return resolved.relative_to(candidate).as_posix()
        except ValueError:
            continue
    return resolved.name


def _read_json(path: Path, context: str) -> tuple[object, str]:
    if not path.is_file():
        raise BenchmarkSourceError(f"missing source file for {context}: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise BenchmarkSourceError(f"cannot read source file for {context}: {path}: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkSourceError(f"malformed JSON for {context} at {path}: {exc}") from exc
    return data, digest


def _require_prompt(value, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkSourceError(f"missing or blank prompt for {context}")
    return value


def _normalise_names(value, *, option: str) -> list[str] | None:
    if value is None:
        return None
    parts = value.split(",") if isinstance(value, str) else list(value)
    names: list[str] = []
    for part in parts:
        name = part.strip()
        if not name:
            raise BenchmarkSourceError(f"{option} contains an empty name")
        if name in names:
            raise BenchmarkSourceError(f"{option} repeats {name!r}")
        names.append(name)
    return names or None


def _selected_t2v_groups(groups) -> list[str]:
    requested = _normalise_names(groups, option="groups")
    if requested is None:
        return list(T2V_CATEGORIES)
    unknown = [group for group in requested if group not in T2V_CATEGORY_SOURCES]
    if unknown:
        raise BenchmarkSourceError(
            f"unknown T2V-CompBench group(s) {unknown}; expected one of {list(T2V_CATEGORIES)}"
        )
    return [group for group in T2V_CATEGORIES if group in requested]


def load_t2v_compbench(
    source_root,
    *,
    groups: Sequence[str] | str | None = None,
    expected_per_category: int | None = T2V_EXPECTED_PER_CATEGORY,
    project_root=None,
) -> list[ManifestRecord]:
    """Load official T2V-CompBench meta_data JSON into manifest records.

    ``source_root`` must contain ``meta_data/<category>.json``.  IDs are
    one-based and zero-padded in each category's original list order.
    """
    root = Path(source_root)
    records: list[ManifestRecord] = []
    for group in _selected_t2v_groups(groups):
        relative = T2V_CATEGORY_SOURCES[group]
        path = root / relative
        data, digest = _read_json(path, f"T2V-CompBench category {group!r}")
        if not isinstance(data, list):
            raise BenchmarkSourceError(
                f"unexpected source structure for T2V-CompBench category {group!r} "
                f"at {path}: expected a JSON list"
            )
        if expected_per_category is not None and len(data) != expected_per_category:
            raise BenchmarkSourceError(
                f"unexpected item count {len(data)} for T2V-CompBench category {group!r} "
                f"in {path}: expected {expected_per_category}"
            )
        source_path = _source_path(path, root, project_root)
        for index, item in enumerate(data):
            context = f"T2V-CompBench category {group!r} item {index}"
            if not isinstance(item, dict):
                raise BenchmarkSourceError(f"unexpected item structure for {context} in {path}")
            prompt = _require_prompt(item.get("prompt"), context)
            records.append(
                ManifestRecord(
                    schema_version=SCHEMA_VERSION,
                    benchmark=T2V_COMPBENCH,
                    group=group,
                    prompt_id=f"{index + 1:04d}",
                    prompt=prompt,
                    source={"path": source_path, "index": index, "sha256": digest},
                    metadata=item,
                )
            )
    validate_records(records)
    return records


def load_vbench(source_root, *, dimensions=None, project_root=None) -> list[ManifestRecord]:
    """Load the original VBench T2V suite from ``vbench/VBench_full_info.json``.

    One manifest record is emitted per source-list item.  ``group`` is the full
    dimension list and ``metadata`` is the complete original item.
    """
    root = Path(source_root)
    path = root / VBENCH_INFO_PATH
    data, digest = _read_json(path, "VBench full info")
    if not isinstance(data, list):
        raise BenchmarkSourceError(
            f"unexpected source structure for VBench at {path}: expected a JSON list"
        )
    source_path = _source_path(path, root, project_root)
    records: list[ManifestRecord] = []
    for index, item in enumerate(data):
        context = f"VBench item {index}"
        if not isinstance(item, dict):
            raise BenchmarkSourceError(f"unexpected item structure for {context} in {path}")
        prompt = _require_prompt(item.get("prompt_en"), context)
        dimension = item.get("dimension")
        if (
            not isinstance(dimension, list)
            or not dimension
            or any(not isinstance(name, str) or not name.strip() for name in dimension)
        ):
            raise BenchmarkSourceError(
                f"invalid dimension list for {context} in {path}: expected a non-empty list of strings"
            )
        records.append(
            ManifestRecord(
                schema_version=SCHEMA_VERSION,
                benchmark=VBENCH,
                group=tuple(dimension),
                prompt_id=f"{index + 1:04d}",
                prompt=prompt,
                source={"path": source_path, "index": index, "sha256": digest},
                metadata=item,
            )
        )
    validate_records(records)
    if dimensions is not None:
        records = filter_vbench_dimensions(records, dimensions)
        validate_records(records)
    return records


def filter_vbench_dimensions(records: Sequence[ManifestRecord], dimensions) -> list[ManifestRecord]:
    """Keep records whose full dimension list contains any requested dimension."""
    requested = _normalise_names(dimensions, option="dimensions")
    if requested is None:
        raise BenchmarkSourceError("dimensions filter must not be empty")
    available = {name for record in records for name in record.group}
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise BenchmarkSourceError(
            f"unknown VBench dimension(s) {unknown}; available dimensions: {sorted(available)}"
        )
    wanted = set(requested)
    selected = [record for record in records if wanted.intersection(record.group)]
    if not selected:
        raise BenchmarkSourceError(f"no VBench records match dimensions {requested}")
    return selected


def apply_filters(
    records: Sequence[ManifestRecord],
    *,
    ids=None,
    limit: int | None = None,
    per_group_limit: int | None = None,
) -> list[ManifestRecord]:
    """Apply ID, global-limit, and per-group-limit filters in a stable order."""
    selected = list(records)
    wanted_ids = _normalise_names(ids, option="ids")
    if wanted_ids is not None:
        available = {record.prompt_id for record in selected}
        missing = [prompt_id for prompt_id in wanted_ids if prompt_id not in available]
        if missing:
            preview = sorted(available)[:12]
            more = "" if len(available) <= 12 else f", ... ({len(available)} total)"
            raise BenchmarkSourceError(
                f"unknown prompt ID(s) {missing}; available IDs in the current selection: "
                f"{preview}{more}"
            )
        wanted = set(wanted_ids)
        selected = [record for record in selected if record.prompt_id in wanted]
    if per_group_limit is not None:
        if per_group_limit < 1:
            raise BenchmarkSourceError("per_group_limit must be at least 1")
        counts: dict[object, int] = {}
        capped: list[ManifestRecord] = []
        for record in selected:
            key = record.group
            counts[key] = counts.get(key, 0) + 1
            if counts[key] <= per_group_limit:
                capped.append(record)
        selected = capped
    if limit is not None:
        if limit < 1:
            raise BenchmarkSourceError("limit must be at least 1")
        selected = selected[:limit]
    if not selected:
        raise BenchmarkSourceError("no manifest records selected")
    validate_records(selected)
    return selected


def validate_records(records: Sequence[ManifestRecord]) -> None:
    """Validate uniqueness and serializability of a record sequence."""
    if not records:
        raise BenchmarkSourceError("manifest must contain at least one record")
    seen: set[tuple] = set()
    for record in records:
        if not isinstance(record, ManifestRecord):
            raise TypeError(f"expected ManifestRecord, got {type(record).__name__}")
        key = (record.benchmark, record.group, record.prompt_id)
        if key in seen:
            raise BenchmarkSourceError(
                f"duplicate prompt_id {record.prompt_id!r} in namespace "
                f"({record.benchmark}, {record.group})"
            )
        seen.add(key)
        try:
            json.dumps(record.to_dict(), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise BenchmarkSourceError(f"record is not JSON serializable: {exc}") from exc


def write_manifest(records: Sequence[ManifestRecord], output, *, overwrite: bool = False) -> Path:
    """Write UTF-8 JSONL atomically, refusing an existing file by default."""
    validate_records(records)
    destination = Path(output)
    if destination.exists() and not overwrite:
        raise BenchmarkSourceError(
            f"refusing to overwrite existing manifest {destination}; pass --overwrite to replace it"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        handle, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
        )
        with open(handle, "w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        Path(temporary).replace(destination)
        temporary = None
        return destination
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except OSError:
                pass


def manifest_summary(
    records: Sequence[ManifestRecord], benchmark: str, output, source_count: int | None = None
) -> dict:
    """Build the compact summary printed by the CLI."""
    groups: list[object] = []
    seen_groups: set[str] = set()
    sources: dict[str, str] = {}
    for record in records:
        group = list(record.group) if isinstance(record.group, tuple) else record.group
        key = json.dumps(group, ensure_ascii=False)
        if key not in seen_groups:
            seen_groups.add(key)
            groups.append(group)
        sources[record.source["path"]] = record.source["sha256"]
    return {
        "benchmark": benchmark,
        "groups": groups,
        "source_count": len(sources) if source_count is None else source_count,
        "written_count": len(records),
        "output": str(output),
        "source_sha256": sources,
    }
