#!/usr/bin/env python3
"""
Encodes every unique RGB frame referenced in a temporal_dataset.jsonl
manifest (see build_temporal_dataset.py) using DINOv3, caching results to
avoid re-encoding the same image multiple times -- windows overlap
heavily at --stride 1 (window_size=8 means ~8x redundancy across
consecutive examples), so deduplication matters a lot here.

Requires:
    pip install "transformers>=4.56.0" torch pillow
    huggingface-cli login   # DINOv3 checkpoints are gated -- accept the
                             # license on the model's HF page first, then
                             # authenticate, or this will fail to download.

Uses the model's pooled ([CLS]-token) global embedding per image, not the
dense per-patch feature map -- the right default for a single fixed-size
vector per frame feeding a downstream regression/classification probe.
If you specifically want dense patch features instead, this script would
need adapting (outputs.last_hidden_state instead of outputs.pooler_output).

Usage:
    python encode_frames.py \\
        --dataset ./rulebook_dataset_merged \\
        --temporal-manifest ./temporal_dataset.jsonl \\
        --model facebook/dinov3-vitb16-pretrain-lvd1689m \\
        --output ./frame_embeddings.npz
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   help="Dataset root that frame_window paths are relative to (e.g. ./rulebook_dataset_merged)")
    p.add_argument("--temporal-manifest", required=True,
                   help="Output of build_temporal_dataset.py")
    p.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m",
                   help="HF model ID. Smaller/faster: facebook/dinov3-vits16-pretrain-lvd1689m. "
                        "Larger/slower: facebook/dinov3-vitl16-pretrain-lvd1689m")
    p.add_argument("--output", default="./frame_embeddings.npz")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default=None,
                   help="cuda / cpu / mps. Defaults to cuda if available, else cpu.")
    return p.parse_args()


def collect_unique_paths(manifest_path):
    paths = set()
    with open(manifest_path) as f:
        for line in f:
            row = json.loads(line)
            paths.update(row["frame_window"])
            # rgb_frame is the current frame; already covered when the
            # window includes the current frame, but harmless to include
            # even in --exclude-current manifests (encoding a frame that
            # ends up unused just means one extra cached row, not a bug).
            paths.add(row["rgb_frame"])
    return sorted(paths)


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    paths = collect_unique_paths(args.temporal_manifest)
    print(f"Found {len(paths)} unique frame(s) to encode (device={device}, model={args.model})")

    print("Loading DINOv3 processor + model (may download on first run)...")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    all_embeddings = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(paths), args.batch_size)):
            batch_paths = paths[i:i + args.batch_size]
            images = []
            for rel_path in batch_paths:
                full_path = os.path.join(args.dataset, rel_path)
                images.append(Image.open(full_path).convert("RGB"))

            inputs = processor(images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            pooled = outputs.pooler_output.float().cpu().numpy()  # (batch, hidden_dim)
            all_embeddings.append(pooled)

            done = min(i + args.batch_size, len(paths))

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Final embedding matrix: {embeddings.shape}")

    np.savez_compressed(args.output, embeddings=embeddings, paths=np.array(paths))
    print(f"Saved embedding cache to {args.output}")


if __name__ == "__main__":
    main()