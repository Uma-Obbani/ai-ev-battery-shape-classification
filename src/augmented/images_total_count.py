from pathlib import Path

# Processed folder
OUT_DIR = Path("/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/processed/train")


# Recursively find all JPG images
images = list(OUT_DIR.rglob("*.jpg"))
print(f"Total images in processed TRAIN folder: {len(images)}")

