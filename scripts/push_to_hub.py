import argparse
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a local dataset folder to Hugging Face Hub.")
    parser.add_argument("--root", type=str, required=True, help="Local dataset root path (e.g. current_data/my_episode)")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo ID (e.g. user/dataset_name)")
    parser.add_argument("--private", action="store_true", help="Make the dataset private on the Hub")
    parser.add_argument("--branch", default=None, help="Optional branch to upload to.")
    parser.add_argument("--large_folder", action="store_true", help="Use upload_large_folder for large datasets.")
    args = parser.parse_args()

    root_path = Path(args.root)
    if not root_path.exists():
        print(f"[ERROR] Local root path does not exist: {args.root}")
        return
    if not root_path.is_dir():
        print(f"[ERROR] Local root path is not a directory: {args.root}")
        return

    print(f"[INFO] Uploading dataset folder: {root_path}")
    print(f"[INFO] Target Hugging Face repo: {args.repo_id} (private={args.private})")
    try:
        api = HfApi()
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        if args.branch:
            api.create_branch(
                repo_id=args.repo_id,
                repo_type="dataset",
                branch=args.branch,
                exist_ok=True,
            )

        upload_kwargs: dict[str, Any] = {
            "repo_id": args.repo_id,
            "repo_type": "dataset",
            "folder_path": root_path,
            "revision": args.branch,
            "ignore_patterns": ["images/"],
        }
        if args.large_folder:
            api.upload_large_folder(**upload_kwargs)
        else:
            api.upload_folder(**upload_kwargs)
        print("[INFO] Successfully uploaded to Hugging Face Hub.")
    except Exception as e:
        print(f"[ERROR] Failed to push to hub: {e}")


if __name__ == "__main__":
    main()
