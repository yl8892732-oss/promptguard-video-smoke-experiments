"""Download one copy of the exact tested model revision; no datasets."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="model")
    parser.add_argument("--revision", default="1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01")
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(Path(".hf").resolve()))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download
    path = snapshot_download("THUDM/CogVideoX-2b", revision=args.revision,
        local_dir=args.directory, max_workers=3,
        allow_patterns=["model_index.json", "scheduler/*", "text_encoder/*.json",
            "text_encoder/*.safetensors", "tokenizer/*", "transformer/*.json",
            "transformer/*.safetensors", "vae/*.json", "vae/*.safetensors"])
    print(path)


if __name__ == "__main__":
    main()
