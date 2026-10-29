"""Plan future PromptGuard generation runs without importing a model.

The planner validates a Stage 03 JSONL manifest, runs the lightweight semantic
parser once per prompt, and writes a deterministic plan.  It never imports
torch/diffusers/transformers, initializes CUDA, starts a subprocess, or writes
an MP4; the actual one-process-per-record runner belongs to a later stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from promptguard.benchmarks import ManifestRecord, VBENCH
from promptguard.semantics import parse_prompt

PLAN_SCHEMA_VERSION = 1
DEFAULT_SEED = 42
REQUIRED_KEYS = {
    "schema_version",
    "benchmark",
    "group",
    "prompt_id",
    "prompt",
    "source",
    "metadata",
}
_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]")


class PlannerError(ValueError):
    """Raised for a manifest, semantic, path, or configuration error."""


def _require_dry_run(args) -> None:
    if not args.dry_run:
        raise PlannerError(
            "--dry-run is required: this stage plans only and must never generate video"
        )


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return number


def _nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if number < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a deterministic CPU-only plan for a benchmark manifest."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model", default="model", help="model string only; never resolved")
    parser.add_argument("--mode", choices=["promptguard", "topk", "dense"], default="promptguard")
    parser.add_argument("--steps", type=_positive_int, default=8)
    parser.add_argument("--frames", type=_positive_int, default=9)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--density", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--block-size", type=_positive_int, default=256)
    parser.add_argument("--warmup", type=_nonnegative_int, default=1)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--vram-gib", type=float, default=23.0)
    parser.add_argument("--offload", choices=["model", "sequential"], default="model")
    return parser


def _build_generation_config(args) -> dict:
    if not 0 < args.vram_gib <= 24:
        raise PlannerError("VRAM cap must be in (0, 24] GiB")
    if not 0 < args.density <= 1:
        raise PlannerError("density must be in (0, 1]")
    if not 0 <= args.alpha <= 1 or not 0 <= args.delta <= 1:
        raise PlannerError("alpha and delta must be in [0, 1]")
    if args.frames <= 0 or (args.frames - 1) % 4:
        raise PlannerError("frames must be a positive 4k+1 value")
    if args.steps <= 0:
        raise PlannerError("steps must be positive")
    if args.warmup < 0:
        raise PlannerError("warmup must be non-negative")
    if args.block_size < 1:
        raise PlannerError("block size must be positive")
    if args.mode != "dense" and args.steps <= args.warmup:
        raise PlannerError("steps must exceed warmup so sparse attention would be tested")
    return {
        "model": args.model,
        "mode": args.mode,
        "steps": args.steps,
        "frames": args.frames,
        "seed": args.seed,
        "density": args.density,
        "alpha": args.alpha,
        "delta": args.delta,
        "block_size": args.block_size,
        "warmup": args.warmup,
        "guidance": args.guidance,
        "vram_gib": args.vram_gib,
        "offload": args.offload,
    }


def _manifest_record(data, line_number: int, manifest: Path) -> ManifestRecord:
    location = f"{manifest}:{line_number}"
    if not isinstance(data, dict):
        raise PlannerError(f"{location}: manifest line is not a JSON object")
    prompt_id = data.get("prompt_id")
    identity = f", prompt_id {prompt_id!r}" if isinstance(prompt_id, str) else ""
    keys = set(data)
    if keys != REQUIRED_KEYS:
        missing = sorted(REQUIRED_KEYS - keys)
        extra = sorted(keys - REQUIRED_KEYS)
        raise PlannerError(
            f"{location}{identity}: invalid manifest schema; missing={missing}, extra={extra}"
        )
    try:
        group = data["group"]
        if data["benchmark"] == VBENCH:
            if not isinstance(group, list):
                raise ValueError("VBench group must be a JSON list")
            group = tuple(group)
        record = ManifestRecord(
            schema_version=data["schema_version"],
            benchmark=data["benchmark"],
            group=group,
            prompt_id=data["prompt_id"],
            prompt=data["prompt"],
            source=data["source"],
            metadata=data["metadata"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PlannerError(f"{location}{identity}: invalid manifest record: {exc}") from exc
    return record


def load_manifest(path: Path) -> list[ManifestRecord]:
    if not path.is_file():
        raise PlannerError(f"manifest does not exist: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PlannerError(f"cannot read manifest {path}: {exc}") from exc
    if not lines:
        raise PlannerError(f"manifest is empty: {path}")
    records: list[ManifestRecord] = []
    seen: set[tuple] = set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise PlannerError(f"{path}:{line_number}: empty manifest line")
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PlannerError(f"{path}:{line_number}: malformed JSON: {exc.msg}") from exc
        record = _manifest_record(data, line_number, path)
        key = (record.benchmark, record.group, record.prompt_id)
        if key in seen:
            raise PlannerError(
                f"{path}:{line_number}, prompt_id {record.prompt_id!r}: duplicate logical record "
                f"for benchmark {record.benchmark!r} and group {record.group!r}"
            )
        seen.add(key)
        records.append(record)
    if not records:
        raise PlannerError(f"manifest contains no records: {path}")
    return records


def _safe_component(value: str, kind: str) -> str:
    if not isinstance(value, str) or not value:
        raise PlannerError(f"unsafe {kind} value: {value!r}")
    if (
        value in {".", ".."}
        or value.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:", value)
        or any(char in value for char in ("/", "\\", "\0"))
    ):
        raise PlannerError(f"unsafe {kind} value for a filesystem path: {value!r}")
    cleaned = _SAFE_COMPONENT_RE.sub("_", value)
    if cleaned in {"", ".", ".."}:
        raise PlannerError(f"unsafe {kind} value for a filesystem path: {value!r}")
    return cleaned


def _group_component(record: ManifestRecord) -> str:
    if isinstance(record.group, tuple):
        raw = "__".join(record.group)
    else:
        raw = record.group
    return _safe_component(raw, "group")


def _output_relative_path(record: ManifestRecord) -> Path:
    benchmark = _safe_component(record.benchmark, "benchmark")
    group = _group_component(record)
    prompt_id = _safe_component(record.prompt_id, "prompt_id")
    return Path(benchmark) / group / f"{prompt_id}.mp4"


def _semantic_units(record: ManifestRecord) -> list[dict]:
    try:
        units = parse_prompt(record.prompt)
    except ValueError as exc:
        raise PlannerError(
            f"manifest prompt_id {record.prompt_id!r} ({record.benchmark}/{record.group}): "
            f"semantic parsing failed: {exc}"
        ) from exc
    if not units:
        raise PlannerError(f"semantic parsing produced no units for prompt_id {record.prompt_id!r}")
    return [
        {"label": unit.label, "spans": [[int(start), int(end)] for start, end in unit.spans]}
        for unit in units
    ]


def _future_argv(config: dict, prompt: str, output_path: str) -> list[str]:
    return [
        sys.executable,
        "scripts/generate.py",
        "--model", config["model"],
        "--prompt", prompt,
        "--mode", config["mode"],
        "--steps", str(config["steps"]),
        "--frames", str(config["frames"]),
        "--seed", str(config["seed"]),
        "--density", str(config["density"]),
        "--alpha", str(config["alpha"]),
        "--delta", str(config["delta"]),
        "--block-size", str(config["block_size"]),
        "--warmup", str(config["warmup"]),
        "--guidance", str(config["guidance"]),
        "--vram-gib", str(config["vram_gib"]),
        "--offload", config["offload"],
        "--output", output_path,
    ]


def build_plans(
    records: list[ManifestRecord], output_dir: Path, config: dict
) -> list[dict]:
    plans: list[dict] = []
    seen_paths: set[str] = set()
    for record in records:
        relative = _output_relative_path(record)
        relative_text = relative.as_posix()
        if relative_text in seen_paths:
            raise PlannerError(
                f"duplicate proposed output path {relative_text!r} for prompt_id "
                f"{record.prompt_id!r}"
            )
        seen_paths.add(relative_text)
        if any(part in {"", ".", ".."} for part in relative.parts):
            raise PlannerError(f"unsafe proposed output path: {relative_text!r}")
        output_path = str(output_dir / relative)
        group = list(record.group) if isinstance(record.group, tuple) else record.group
        plans.append(
            {
                "schema_version": PLAN_SCHEMA_VERSION,
                "benchmark": record.benchmark,
                "group": group,
                "prompt_id": record.prompt_id,
                "prompt": record.prompt,
                "source": dict(record.source),
                "metadata": record.metadata,
                "generation": config,
                "seed": config["seed"],
                "semantic_units": _semantic_units(record),
                "output_path": relative_text,
                "argv": _future_argv(config, record.prompt, output_path),
            }
        )
    return plans


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(manifest: Path, plans: list[dict], config: dict, output_dir: Path) -> dict:
    groups: list[object] = []
    seen_groups: set[str] = set()
    prompt_ids: list[str] = []
    for plan in plans:
        key = json.dumps(plan["group"], ensure_ascii=False)
        if key not in seen_groups:
            seen_groups.add(key)
            groups.append(plan["group"])
        prompt_ids.append(plan["prompt_id"])
    return {
        "schema_version": 1,
        "manifest": str(manifest),
        "manifest_sha256": _manifest_sha256(manifest),
        "record_count": len(plans),
        "mode": config["mode"],
        "seed": config["seed"],
        "output_dir": str(output_dir),
        "plan_path": "plan.jsonl",
        "summary_path": "run_summary.json",
        "groups": groups,
        "prompt_ids": prompt_ids,
        "semantic_unit_count": sum(len(plan["semantic_units"]) for plan in plans),
        "dry_run": True,
        "model_operations": 0,
        "subprocess_operations": 0,
        "cuda_operations": 0,
        "video_operations": 0,
    }


def _write_plan_files(plans: list[dict], summary: dict, output_dir: Path, overwrite: bool) -> None:
    plan_path = output_dir / "plan.jsonl"
    summary_path = output_dir / "run_summary.json"
    if not overwrite:
        existing = [str(path) for path in (plan_path, summary_path) if path.exists()]
        if existing:
            raise PlannerError(
                "refusing to overwrite existing dry-run output(s): "
                f"{', '.join(existing)}; pass --overwrite to replace them"
            )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        plan_text = "".join(json.dumps(plan, ensure_ascii=False) + "\n" for plan in plans)
        summary_text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        plan_tmp = output_dir / ".plan.jsonl.tmp"
        summary_tmp = output_dir / ".run_summary.json.tmp"
        try:
            with plan_tmp.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(plan_text)
            with summary_tmp.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(summary_text)
            os.replace(plan_tmp, plan_path)
            os.replace(summary_tmp, summary_path)
        finally:
            for temporary in (plan_tmp, summary_tmp):
                try:
                    if temporary.exists():
                        temporary.unlink()
                except OSError:
                    pass
    except OSError as exc:
        raise PlannerError(f"failed to write dry-run output under {output_dir}: {exc}") from exc


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        _require_dry_run(args)
        config = _build_generation_config(args)
        records = load_manifest(args.manifest)
        plans = build_plans(records, args.output_dir, config)
        summary = _summary(args.manifest, plans, config, args.output_dir)
        _write_plan_files(plans, summary, args.output_dir, args.overwrite)
    except PlannerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        f"Dry run complete: {len(plans)} record(s), seed {config['seed']}; "
        "no model, subprocess, CUDA, or video operation was performed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
