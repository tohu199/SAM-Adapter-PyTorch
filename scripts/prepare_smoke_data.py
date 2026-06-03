"""Create a tiny paired image dataset for pipeline smoke tests."""
import os
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1] / "load" / "codDataset"


def write_pair(img_dir: Path, gt_dir: Path, name: str, color: tuple) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (128, 128), color)
    draw = ImageDraw.Draw(img)
    draw.ellipse((32, 32, 96, 96), fill=(255, 255, 255))
    img.save(img_dir / f"{name}.jpg")

    mask = Image.new("L", (128, 128), 0)
    ImageDraw.Draw(mask).ellipse((32, 32, 96, 96), fill=255)
    mask.save(gt_dir / f"{name}.png")


def main() -> None:
    for split in ("train", "val"):
        for i, color in enumerate(((40, 80, 120), (120, 60, 40))):
            write_pair(
                ROOT / split / "img",
                ROOT / split / "gt",
                f"sample_{i:02d}",
                color,
            )
    print(f"Wrote smoke dataset under {ROOT}")


if __name__ == "__main__":
    main()
