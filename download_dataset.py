"""
Download the AIME dataset from HuggingFace.

The dataset is gated — you must request access at:
    https://huggingface.co/datasets/DAISLab-Unisa/AIME-Dataset

Then set your HF_TOKEN in a .env file (see .env.example) or as an env variable.

Usage:
    python download_dataset.py                  # download everything
    python download_dataset.py --labels-only    # download only CSV label files
"""
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

DATASET_ID = "DAISLab-Unisa/AIME-Dataset"
DEFAULT_LOCAL_DIR = "./dataset"


def main():
    parser = argparse.ArgumentParser(description="Download AIME dataset from HuggingFace")
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR,
                        help="Local directory to download into (default: ./dataset)")
    parser.add_argument("--labels-only", action="store_true",
                        help="Download only metadata/label CSV files (no media)")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN not found. Set it in a .env file or as an environment variable.\n"
            "Get your token at: https://huggingface.co/settings/tokens"
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub not installed. Run: pip install huggingface_hub")

    ignore_patterns = None
    if args.labels_only:
        ignore_patterns = ["data/*"]
        print("Downloading label files only (no media)...")
    else:
        print("Downloading full AIME dataset (images + videos + labels)...")
        print("Note: total size is approximately 2-3 GB. This may take a while.")

    local_path = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=args.local_dir,
        token=token,
        ignore_patterns=ignore_patterns,
    )
    print(f"\nDownload complete: {local_path}")
    print("\nDataset structure:")
    for root, dirs, files in os.walk(local_path):
        # Skip hidden .huggingface cache dirs
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        level = root.replace(local_path, '').count(os.sep)
        indent = '  ' * level
        folder = os.path.basename(root)
        n_files = len(files)
        if n_files > 0:
            print(f"{indent}{folder}/  ({n_files} files)")


if __name__ == "__main__":
    main()
