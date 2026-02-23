#!/usr/bin/env python3
"""
Export CelebA from HuggingFace cache (Arrow) to a directory of image files
so torch-fidelity can use it as reference (--input2).

Usage:
  python scripts/export_celeba_images.py [--cache-dir ./data/celeba] [--output-dir ./data/celeba_images] [--split train]
"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export CelebA cache to image directory for torch-fidelity")
    parser.add_argument("--cache-dir", default="./data/celeba", help="HuggingFace dataset cache (Arrow)")
    parser.add_argument("--output-dir", default="./data/celeba_images", help="Output directory for PNG images")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"], help="Split to export")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to export (default: all)")
    args = parser.parse_args()

    cache_path = Path(args.cache_dir)
    if not cache_path.exists() or not (cache_path / "dataset_dict.json").exists():
        print(f"Error: HuggingFace cache not found at {args.cache_dir}")
        print("Run training or sampling once so the dataset is downloaded/cached.")
        return 1

    from datasets import load_from_disk

    print(f"Loading dataset from {args.cache_dir}...")
    dataset = load_from_disk(args.cache_dir)
    data = list(dataset[args.split])
    if args.max_images is not None:
        data = data[: args.max_images]
    n = len(data)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Exporting {n} images to {args.output_dir}...")
    for idx, item in enumerate(data):
        img = item["image"]
        if not hasattr(img, "save"):
            import numpy as np
            from PIL import Image
            img = Image.fromarray(np.array(img))
        img_path = os.path.join(args.output_dir, f"{idx:06d}.png")
        img.save(img_path)
        if (idx + 1) % 5000 == 0:
            print(f"  {idx + 1}/{n}")
    print(f"Done. Exported {n} images to {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
