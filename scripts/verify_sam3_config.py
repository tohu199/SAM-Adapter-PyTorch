"""Check that a YAML config targets SAM3 (not SAM1/SAM2)."""
import argparse
from pathlib import Path
from typing import Optional

import yaml

SAM3_ENCODER = {
    "inp_size": 1008,
    "img_size": 1008,
    "embed_dim": 1024,
    "depth": 24,
    "num_heads": 16,
    "global_attn_indexes": [5, 11, 17, 23],
}


def verify(config_path: Path, checkpoint_path: Optional[Path]) -> int:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    errors = []
    warnings = []

    ckpt_cfg = str(cfg.get("sam_checkpoint", ""))
    if "sam3" not in ckpt_cfg.lower():
        errors.append(f"sam_checkpoint does not reference sam3: {ckpt_cfg!r}")
    if "sam2" in ckpt_cfg.lower():
        errors.append(f"sam_checkpoint looks like SAM2: {ckpt_cfg!r}")

    model_args = cfg.get("model", {}).get("args", {})
    enc = model_args.get("encoder_mode", {})

    for key, expected in SAM3_ENCODER.items():
        if key in ("inp_size",):
            actual = model_args.get(key)
        else:
            actual = enc.get(key)
        if actual != expected:
            errors.append(f"model encoder mismatch {key}: {actual!r} (expected {expected!r})")

    sam2_keys = ("stages", "backbone_channel_list", "window_spec")
    for key in sam2_keys:
        if key in enc:
            errors.append(f"encoder_mode contains SAM2-only field {key!r}")

    if enc.get("embed_dim") == 768 and enc.get("depth") == 12:
        errors.append("encoder looks like SAM1 ViT-B (768/12), not SAM3 ViT-L (1024/24)")

    if checkpoint_path is not None:
        if not checkpoint_path.is_file():
            warnings.append(f"checkpoint file missing: {checkpoint_path}")
        else:
            import torch

            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(ckpt, dict) and "model" in ckpt:
                ckpt = ckpt["model"]
            keys = list(ckpt.keys())
            detector_keys = [k for k in keys if k.startswith("detector.backbone.")]
            if not detector_keys:
                errors.append(
                    "checkpoint has no detector.backbone.* keys "
                    "(expected official facebook/sam3 sam3.pt format)"
                )
            else:
                print(f"checkpoint OK: {len(detector_keys)} detector.backbone keys")

    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print(f"FAIL: {config_path} is not a valid SAM3 config")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"OK: {config_path} matches SAM3 adapter settings")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    ckpt = args.checkpoint
    if ckpt is None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        ckpt = Path(cfg["sam_checkpoint"])
        if not ckpt.is_absolute():
            ckpt = args.config.resolve().parents[1] / ckpt

    raise SystemExit(verify(args.config, ckpt))


if __name__ == "__main__":
    main()
