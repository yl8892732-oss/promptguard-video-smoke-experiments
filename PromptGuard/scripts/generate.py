"""Run actual pretrained CogVideoX-2B, with a hard process VRAM allocation cap."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import time

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from transformers import T5TokenizerFast

from promptguard import Config
from promptguard.cogvideox import Controller
from promptguard.semantics import SemanticUnit, parse_prompt, align_units


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model", help="local weights directory or HF model ID")
    parser.add_argument("--prompt", default="A small red bird flies above a blue car.")
    parser.add_argument("--units-json", help='JSON list of {"label":..., "spans":[[start,end],...]}')
    parser.add_argument("--mode", choices=["promptguard", "topk", "dense"], default="promptguard")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--density", type=float, default=.4)
    parser.add_argument("--alpha", type=float, default=.2)
    parser.add_argument("--delta", type=float, default=.05)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--guidance", type=float, default=6.)
    parser.add_argument("--vram-gib", type=float, default=23.)
    parser.add_argument("--offload", choices=["model", "sequential"], default="model")
    parser.add_argument("--output", default="results/promptguard.mp4")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Full pretrained verification requires CUDA. Run pytest for CPU tests.")
    if not 0 < args.vram_gib <= 24 or args.steps <= 0 or args.frames <= 0 or (args.frames-1) % 4:
        raise SystemExit("VRAM cap must be in (0,24] GiB, steps positive, frames=4k+1")
    if args.mode != "dense" and args.steps <= args.warmup:
        raise SystemExit("steps must exceed warmup so that sparse attention is actually tested")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(args.vram_gib * 2**30 / total, .98), 0)
    torch.cuda.reset_peak_memory_stats()
    config = Config(args.density, 0 if args.mode == "topk" else args.alpha,
                    0 if args.mode == "topk" else args.delta, args.block_size, args.warmup)
    started = time.perf_counter()
    pipe = CogVideoXPipeline.from_pretrained(args.model, torch_dtype=torch.float16,
                                             local_files_only=Path(args.model).is_dir())
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    if args.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()
    units = [SemanticUnit(x["label"], tuple(tuple(span) for span in x["spans"])) for x in json.loads(Path(args.units_json).read_text())] if args.units_json else parse_prompt(args.prompt)
    fast = T5TokenizerFast.from_pretrained(args.model, subfolder="tokenizer",
                                           local_files_only=Path(args.model).is_dir())
    expected = pipe.tokenizer(args.prompt, padding="max_length", max_length=226,
                             truncation=True, add_special_tokens=True)["input_ids"]
    tokens = align_units(args.prompt, units, fast, 226, expected_ids=expected)
    print("Semantic units:", [(u.label, indices) for u, indices in zip(units, tokens)], flush=True)
    controller = Controller(pipe.transformer, tokens, config, args.guidance > 1) if args.mode != "dense" else None
    per_step = []
    last = time.perf_counter()
    def callback(pipeline, i, timestep, kwargs):
        nonlocal last
        torch.cuda.synchronize()
        now = time.perf_counter()
        record = dict(step=i, seconds=now-last, allocated_gib=torch.cuda.memory_allocated()/2**30)
        per_step.append(record)
        print(json.dumps(record), flush=True)
        last = now
        if not torch.isfinite(kwargs["latents"]).all():
            raise RuntimeError(f"non-finite latents at step {i}")
        return controller.callback(pipeline, i, timestep, kwargs) if controller else kwargs
    inference_start = time.perf_counter()
    with torch.inference_mode():
        video = pipe(prompt=args.prompt, height=480, width=720, num_frames=args.frames,
                     num_inference_steps=args.steps, guidance_scale=args.guidance,
                     generator=torch.Generator(device="cuda").manual_seed(args.seed),
                     callback_on_step_end=callback, output_type="np").frames[0]
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start
    import numpy as np
    if not np.isfinite(video).all() or float(video.std()) < 1e-5:
        raise RuntimeError("decoded video is invalid or constant")
    export_to_video(video, str(output), fps=8)
    import diffusers, transformers
    report = dict(model=args.model, mode=args.mode, prompt=args.prompt, config=asdict(config),
        seed=args.seed, frames=args.frames, steps=args.steps, guidance=args.guidance,
        resolution=[720, 480], dtype="float16", offload=args.offload,
        gpu=torch.cuda.get_device_name(), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        vram_cap_gib=args.vram_gib, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        elapsed_seconds=time.perf_counter()-started, inference_seconds=inference_seconds,
        video_shape=list(video.shape), pixel_std=float(video.std()), finite=True,
        versions=dict(torch=torch.__version__, diffusers=diffusers.__version__, transformers=transformers.__version__, python=platform.python_version()),
        semantic_units=[dict(label=u.label, spans=u.spans, tokens=t) for u, t in zip(units, tokens)],
        step_metrics=per_step, routing=controller.report() if controller else {})
    assert report["peak_reserved_gib"] <= args.vram_gib + .1
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"routing", "step_metrics", "semantic_units"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
