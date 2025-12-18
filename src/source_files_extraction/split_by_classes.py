from pathlib import Path
import shutil
import random

SUFFIX_MAP = {
    "_cyl": "cylindrical",
    "_po": "pouch",
    "_pri": "prismatic",
}

RAW_DIR = Path("/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/zenodo_raw_files/recybat24_aug/recybat24-aug")
OUT_DIR = Path("/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classificationdata/Augumented")
"""
print("TRAIN exists:", (RAW_DIR / "train").exists())
print("VAL exists:", (RAW_DIR / "val").exists())


images = list((RAW_DIR / "train").rglob("*.jpg"))
print(f"Total images in train folder: {len(images)}")

images = list((RAW_DIR / "val").rglob("*.jpg"))
print(f"Total images in val folder: {len(images)}")
"""

random.seed(42)

def get_class(filename: str):
    filename = filename.lower()
    if "_pri_" in filename:
        return "prismatic"
    elif "_cyl_" in filename:
        return "cylindrical"
    elif "_po_" in filename:
        return "pouch"
    return None

def move_group(images, base_dst, label):
    moved = 0
    for img in images:
        cls = get_class(img.name)
        if not cls:
            continue

        target = base_dst / cls
        target.mkdir(parents=True, exist_ok=True)

        shutil.move(img, target / img.name)
        moved += 1

    print(f"✅ {label}: moved {moved} images")

def split_train():
    src = RAW_DIR / "train"
    dst = OUT_DIR / "train"
    dst.mkdir(parents=True, exist_ok=True)

    images = list(src.rglob("*.jpg"))  # recursive search
    move_group(images, dst, "train")

def split_val_test():
    src = RAW_DIR / "val"
    val_dst = OUT_DIR / "val"
    test_dst = OUT_DIR / "test"

    val_dst.mkdir(parents=True, exist_ok=True)
    test_dst.mkdir(parents=True, exist_ok=True)

    images = list(src.rglob("*.jpg"))  # recursive search
    random.shuffle(images)

    split_idx = len(images) // 2
    val_images = images[:split_idx]
    test_images = images[split_idx:]

    move_group(val_images, val_dst, "val")
    move_group(test_images, test_dst, "test")

if __name__ == "__main__":
    split_train()
    split_val_test()
