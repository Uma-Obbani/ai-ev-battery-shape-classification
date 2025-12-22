import tarfile
import os

# Path to your tar.gz file
tar_file = "/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/zenodo_raw_files/recybat24_aug.tar.gz"

# Path where you want to extract
extract_dir = "/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/zenodo_raw_files/recybat24_aug"

# Make sure extraction folder exists
os.makedirs(extract_dir, exist_ok=True)

# Extract the tar.gz file
with tarfile.open(tar_file, 'r:gz') as tar_ref:
    tar_ref.extractall(extract_dir)

print("Extraction complete!")
print("Extracted files:", os.listdir(extract_dir))

