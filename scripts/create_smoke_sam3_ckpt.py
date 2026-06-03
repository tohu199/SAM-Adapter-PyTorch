"""
Build a sam3.pt-shaped checkpoint for pipeline smoke tests only.

Uses the adapter model's image_encoder weights under detector.backbone.*
so train.py's SAM3 loader can run without Hugging Face access.
Replace with scripts/download_sam3_ckpt.py for real training.
"""
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import models  # noqa: E402


def main() -> None:
    with open(ROOT / "configs" / "cod-sam-vit-l.yaml") as f:
        cfg = yaml.safe_load(f)

    model = models.make(cfg["model"])
    state = model.state_dict()
    ckpt = {}

    for key, value in state.items():
        if key.startswith("image_encoder."):
            ckpt["detector.backbone." + key[len("image_encoder.") :]] = value.cpu()
        elif key.startswith("mask_decoder."):
            ckpt[key] = value.cpu()
        elif key.startswith("pe_layer."):
            ckpt[key] = value.cpu()
        elif key == "no_mask_embed.weight":
            ckpt[key] = value.cpu()

    out = ROOT / "pretrained" / "sam3.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": ckpt}, out)
    print(f"Wrote smoke checkpoint ({len(ckpt)} tensors) to {out}")


if __name__ == "__main__":
    main()
