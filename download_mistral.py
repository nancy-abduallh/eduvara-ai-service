# download_mistral_fixed.py
import os
import ssl
import certifi
import urllib3
import requests
from tqdm import tqdm
import sys

# Fix SSL issues
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["CURL_CA_BUNDLE"] = certifi.where()

# Disable SSL warnings (for troubleshooting)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set token
HF_TOKEN = "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"

# Model files to download (Mistral-7B-Instruct-v0.2 - correct file names)
files = [
    "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/config.json",
    "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/tokenizer.json",
    "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/tokenizer_config.json",
    "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/model-00001-of-00003.safetensors",
    "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/model-00002-of-00003.safetensors",
    "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/model-00003-of-00003.safetensors",
    "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/model.safetensors.index.json",
]

local_dir = "./models/Mistral-7B-Instruct"
os.makedirs(local_dir, exist_ok=True)

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Create a session with custom SSL context
session = requests.Session()
session.verify = certifi.where()

for url in files:
    filename = url.split("/")[-1]
    filepath = os.path.join(local_dir, filename)
    
    # Check if file exists and has size > 0
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000000:  # > 1MB
        print(f"✅ {filename} already exists ({(os.path.getsize(filepath)/1e9):.2f} GB), skipping...")
        continue
    
    print(f"Downloading {filename}...")
    
    try:
        response = session.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        print(f"   Size: {total_size/(1024**3):.2f} GB")
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        print(f"✅ Downloaded {filename}")
        
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        print(f"   Trying alternative method...")
        
        # Alternative: Use wget style with verify=False
        try:
            response = session.get(url, headers=headers, stream=True, verify=False, timeout=60)
            total_size = int(response.headers.get('content-length', 0))
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"{filename} (no SSL)") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            print(f"✅ Downloaded {filename} (with SSL verification disabled)")
        except Exception as e2:
            print(f"❌ Alternative also failed: {e2}")

print("\n" + "=" * 60)
print("Download complete! Checking files...")
print("=" * 60)

# Verify files
for url in files:
    filename = url.split("/")[-1]
    filepath = os.path.join(local_dir, filename)
    if os.path.exists(filepath):
        size_gb = os.path.getsize(filepath) / (1024**3)
        print(f"✅ {filename}: {size_gb:.2f} GB")
    else:
        print(f"❌ {filename}: MISSING")

print("\n✅ Script finished!")