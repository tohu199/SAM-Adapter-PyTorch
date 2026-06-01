#!/usr/bin/env python3
"""Generate a toy RGB dataset: yellow circles (class 1) and red squares (class 2) on black."""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Mask colors (must match datasets/color_utils.py PALETTE)
MASK_BG = (0, 0, 0)
MASK_YELLOW = (255, 255, 0)
MASK_RED = (255, 0, 0)

# Image draw colors (visual only)
IMG_BG = (235, 235, 235)
IMG_YELLOW = (220, 200, 0)
IMG_RED = (200, 40, 40)


def _rand_bbox(rng, size, margin=20, max_side=90):
    side = int(rng.randint(40, max_side + 1))
    x0 = int(rng.randint(margin, size - side - margin))
    y0 = int(rng.randint(margin, size - side - margin))
    return x0, y0, x0 + side, y0 + side


def make_sample(seed, size=256):
    rng = np.random.RandomState(seed)
    img = Image.new('RGB', (size, size), IMG_BG)
    mask = Image.new('RGB', (size, size), MASK_BG)
    d_img = ImageDraw.Draw(img)
    d_mask = ImageDraw.Draw(mask)

    n_shapes = int(rng.randint(1, 5))
    for _ in range(n_shapes):
        is_circle = rng.rand() < 0.5
        x0, y0, x1, y1 = _rand_bbox(rng, size)
        if is_circle:
            d_img.ellipse([x0, y0, x1, y1], fill=IMG_YELLOW, outline=IMG_YELLOW)
            d_mask.ellipse([x0, y0, x1, y1], fill=MASK_YELLOW, outline=MASK_YELLOW)
        else:
            d_img.rectangle([x0, y0, x1, y1], fill=IMG_RED, outline=IMG_RED)
            d_mask.rectangle([x0, y0, x1, y1], fill=MASK_RED, outline=MASK_RED)

    return img, mask


def write_split(out_root: Path, split: str, count: int, start_seed: int, size: int):
    img_dir = out_root / 'images' / split
    mask_dir = out_root / 'masks' / split
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        seed = start_seed + i
        img, mask = make_sample(seed, size=size)
        name = f'sample_{i:04d}.png'
        img.save(img_dir / name)
        mask.save(mask_dir / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='./load/shapes_demo')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--train-count', type=int, default=40)
    parser.add_argument('--val-count', type=int, default=10)
    args = parser.parse_args()

    out_root = Path(args.out)
    write_split(out_root, 'train', args.train_count, start_seed=0, size=args.size)
    write_split(out_root, 'val', args.val_count, start_seed=10_000, size=args.size)
    print(f'Wrote {args.train_count} train + {args.val_count} val samples to {out_root.resolve()}')


if __name__ == '__main__':
    main()
