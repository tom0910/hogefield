# delete this comment
import os
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

print("Starting download of Google Speech Commands dataset...")

data_dir = os.path.join("/project/", "data", "GSC")

# Create directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

dataset = SPEECHCOMMANDS(root=data_dir, download=True)

print("Download complete.")

# usage:
# python scripts/download_data.py