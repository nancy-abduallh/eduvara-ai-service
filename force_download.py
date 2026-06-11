# force_download.py
import os
os.environ["HF_TOKEN"] = "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"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download
import sys

print("Starting download of Qwen2-7B-Instruct (15 GB)...")
print("This will take 20-30 minutes. Please wait...")

try:
    snapshot_download(
        repo_id="Qwen/Qwen2-7B-Instruct",
        local_dir="./models/Qwen2-7B-Instruct",
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4
    )
    print("✅ Download complete!")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)