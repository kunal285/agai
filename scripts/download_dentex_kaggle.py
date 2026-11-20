"""Download and organize DENTEX dataset directly from Kaggle via kagglehub.

Dataset: truthisneverlinear/dentex-challenge-2023
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import kagglehub
from dentistry.config import load_config
from scripts.organize_dentex import organize_dentex

def download_and_setup_dentex(target_dir: str | Path | None = None):
    cfg = load_config()
    target = Path(target_dir or cfg["paths"]["raw_data_dir"]).resolve()

    print("Initiating DENTEX download via kagglehub...")
    print("Dataset: truthisneverlinear/dentex-challenge-2023")
    
    # Download dataset files to kagglehub cache directory
    download_path = kagglehub.dataset_download("truthisneverlinear/dentex-challenge-2023")
    print(f"\nDownload completed / verified at cache path: {download_path}")

    # Inspect contents
    cache_path = Path(download_path)
    all_files = list(cache_path.rglob("*"))
    print(f"Total files in downloaded dataset: {len(all_files)}")

    # Automatically organize into data/DENTEX
    print(f"\nOrganizing files into canonical workspace path: {target}...")
    summary = organize_dentex(cache_path, target)
    print("\nDataset successfully downloaded and organized into workspace!")
    return summary

def main():
    parser = argparse.ArgumentParser(description="Download DENTEX dataset using kagglehub and organize into project data dir.")
    parser.add_argument("--target-dir", default=None, help="Target directory (default: from configs/default.yaml)")
    args = parser.parse_args()

    download_and_setup_dentex(args.target_dir)

if __name__ == "__main__":
    main()
