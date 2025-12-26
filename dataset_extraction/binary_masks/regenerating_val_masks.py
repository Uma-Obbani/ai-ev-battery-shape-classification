from pycocotools.coco import COCO
import os
import cv2
import numpy as np
from tqdm import tqdm

# ---------------- CONFIG ----------------
VAL_JSON = "/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/raw/val/instance_segmentation_annotations.json"
VAL_IMAGE_DIR = "/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/processed/val/images"
VAL_MASK_DIR  = "/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/processed/val/masks"

# Folder mapping based on filename code (LOWERCASE)
CODE_TO_FOLDER = {
    "cyl": "cylindrical",
    "po":  "pouch",
    "pri": "prismatic"
}
# ---------------------------------------

def collect_images(root):
    imgs = {}
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                imgs[os.path.splitext(f)[0].lower()] = f
    return imgs

def infer_category(filename):
    """
    Infer category from filename using code inside filename
    """
    fname_lower = filename.lower()
    for code, folder in CODE_TO_FOLDER.items():
        if code in fname_lower:
            return folder
    return None

# ---------------- RUN ----------------
coco = COCO(VAL_JSON)
images = collect_images(VAL_IMAGE_DIR)

# Map image name → COCO id
coco_map = {
    os.path.splitext(img["file_name"])[0].lower(): img["id"]
    for img in coco.dataset["images"]
}

# Ensure mask subfolders exist
for folder in CODE_TO_FOLDER.values():
    os.makedirs(os.path.join(VAL_MASK_DIR, folder), exist_ok=True)

count = 0
skipped = 0

for name, original_file in tqdm(images.items(), desc="Rebuilding validation masks"):
    img_id = coco_map.get(name)
    if img_id is None:
        skipped += 1
        continue

    category_folder = infer_category(original_file)
    if category_folder is None:
        print(f"⚠️ Unknown category for {original_file}")
        skipped += 1
        continue

    img_info = coco.loadImgs(img_id)[0]
    mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)

    for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
        if ann.get("iscrowd", 0):
            continue
        mask[coco.annToMask(ann) > 0] = 255

    mask_path = os.path.join(
        VAL_MASK_DIR,
        category_folder,
        name + ".png"
    )

    cv2.imwrite(mask_path, mask)
    count += 1

print("\n✅ Validation masks regenerated correctly")
print(f"Saved masks : {count}")
print(f"Skipped     : {skipped}")
