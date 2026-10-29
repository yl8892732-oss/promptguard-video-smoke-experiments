"""Build a unified JSONL manifest from official T2V-CompBench or VBench prompts.

This CLI only reads prompt/metadata JSON.  It never imports torch, initializes
CUDA, parses prompt semantics, downloads a model, or generates a video.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from promptguard.benchmarks import (
    T2V_CATEGORIES,
    T2V_COMPBENCH,
    VBENCH,
    BenchmarkSourceError,
    apply_filters,
    load_t2v_compbench,
    load_vbench,
    manifest_summary,
    write_manifest,
)


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from exc
    if number < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic JSONL manifest from official benchmark prompt metadata."
    )
    parser.add_argument("--benchmark", required=True, choices=[T2V_COMPBENCH, VBENCH])
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="T2V root containing meta_data/, or VBench root containing vbench/",
    )
    parser.add_argument("--output", required=True, type=Path, help="destination JSONL file")
    parser.add_argument("--limit", type=_positive_int, help="keep the first N records after other filters")
    parser.add_argument("--ids", help="comma-separated prompt IDs to keep")
    parser.add_argument(
        "--groups",
        help=f"T2V-CompBench only: comma-separated categories from {list(T2V_CATEGORIES)}",
    )
    parser.add_argument(
        "--dimensions",
        help="VBench only: comma-separated dimension names; keeps records containing any of them",
    )
    parser.add_argument(
        "--per-group-limit",
        type=_positive_int,
        help="keep the first N records per group (T2V category or VBench dimension list)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the output file if it already exists",
    )
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.benchmark == VBENCH:
            if args.groups is not None:
                raise BenchmarkSourceError("--groups is T2V-CompBench-only; use --dimensions for VBench")
            records = load_vbench(args.source_root, dimensions=args.dimensions)
        else:
            if args.dimensions is not None:
                raise BenchmarkSourceError(
                    "--dimensions is VBench-only; use --groups for T2V-CompBench"
                )
            records = load_t2v_compbench(args.source_root, groups=args.groups)
        records = apply_filters(
            records,
            ids=args.ids,
            limit=args.limit,
            per_group_limit=args.per_group_limit,
        )
        write_manifest(records, args.output, overwrite=args.overwrite)
        summary = manifest_summary(records, args.benchmark, args.output)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except BenchmarkSourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
