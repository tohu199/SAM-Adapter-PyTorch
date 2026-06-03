"""Verify SAM3-specific code paths without starting training."""
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import models  # noqa: E402


def load_config():
    with open(ROOT / "configs" / "cod-sam-vit-l.yaml") as f:
        return yaml.safe_load(f)


def main() -> int:
    cfg = load_config()
    model_spec = {
        "name": cfg["model"]["name"],
        "args": cfg["model"]["args"],
    }
    model = models.make(model_spec).cpu()
    model.device = torch.device("cpu")
    enc = cfg["model"]["args"]["encoder_mode"]
    assert enc["embed_dim"] == 1024 and enc["depth"] == 24
    assert cfg["model"]["args"]["inp_size"] == 1008

    train_py = (ROOT / "train.py").read_text()
    assert "detector.backbone." in train_py

    x = torch.randn(1, 3, 1008, 1008)
    gt = torch.randn(1, 1, 1008, 1008)
    model.set_input(x, gt)
    with torch.no_grad():
        model.forward()
    assert model.pred_mask.shape == (1, 1, 1008, 1008)

    print("SAM3 codebase checks passed:")
    print(f"  - ViT-L adapter model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print(f"  - forward @ 1008: output {tuple(model.pred_mask.shape)}")
    print("  - train.py maps detector.backbone.* -> image_encoder.*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
