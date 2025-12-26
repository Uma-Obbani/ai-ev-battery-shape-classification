import os
import shutil

VAL_MASK_DIR = "/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/processed/val/masks"

print("🧹 Cleaning validation mask folder...")

for item in os.listdir(VAL_MASK_DIR):
    path = os.path.join(VAL_MASK_DIR, item)
    if os.path.isdir(path):
        shutil.rmtree(path)   # remove old category folders
    else:
        os.remove(path)       # remove old mask files

print("✅ Validation masks cleaned completely")
