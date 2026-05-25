import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


BASE_DIR = Path("checkpoints")


def download_model(model_id: str) -> None:
    model_name = model_id.split("/")[-1]
    target_dir = BASE_DIR / model_name
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model from: {model_id}")
    snapshot_download(repo_id=model_id, local_dir=target_dir)
    print(f"Model downloaded to: {target_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download an OpenVLA-OFT checkpoint locally.")
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID")
    args = parser.parse_args()
    download_model(args.model_id)
