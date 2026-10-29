"""Execute generate.py and independently sample the process's NVIDIA VRAM.

Run with CUDA_VISIBLE_DEVICES set to one physical GPU. The report uses the
process ID, so unrelated users' jobs are never counted or modified.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="results/nvidia_memory.json")
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([sys.executable, "scripts/generate.py", *forwarded], env=os.environ.copy())
    samples = []
    errors = 0
    try:
        while process.poll() is None:
            try:
                result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
                for line in result.stdout.splitlines():
                    pid, memory = [x.strip() for x in line.split(",")]
                    if int(pid) == process.pid:
                        samples.append(float(memory))
            except (subprocess.SubprocessError, ValueError):
                errors += 1
            time.sleep(.5)
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise
    report = dict(pid=process.pid, returncode=process.returncode, samples=len(samples),
                  nvidia_smi_peak_mib=max(samples, default=0),
                  nvidia_smi_peak_gib=max(samples, default=0)/1024,
                  poll_errors=errors, sampling_interval_seconds=.5,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if process.returncode:
        raise SystemExit(process.returncode)
    if not samples or report["nvidia_smi_peak_gib"] > 24:
        raise SystemExit("VRAM verification failed or exceeded 24 GiB")


if __name__ == "__main__":
    main()
