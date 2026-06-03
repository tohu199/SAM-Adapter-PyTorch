"""Download SAM3 checkpoint from Hugging Face (gated; requires HF login)."""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "pretrained" / "sam3.pt",
    )
    parser.add_argument("--repo", default="facebook/sam3")
    parser.add_argument("--filename", default="sam3.pt")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = hf_hub_download(repo_id=args.repo, filename=args.filename)
    shutil.copy2(ckpt_path, args.out)
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
