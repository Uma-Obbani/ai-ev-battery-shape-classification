import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

# ================= CONFIG =================
TRAIN_JSON = "/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/raw/train/instance_segmentation_annotations.json"
IMAGE_DIR = "/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/processed/train/images"
MASK_DIR = "/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/processed/train/masks"
# =========================================

# Load COCO annotations
coco = COCO(TRAIN_JSON)

# Map from filename without extension to image ID (robust matching)
filename_to_id = {os.path.splitext(img["file_name"])[0].lower(): img["id"] for img in coco.dataset["images"]}

# Process each subfolder (cylindrical, pouch, prismatic)
for subfolder in os.listdir(IMAGE_DIR):
    subfolder_clean = subfolder.strip()  # remove hidden chars or spaces
    image_subdir = os.path.join(IMAGE_DIR, subfolder_clean)
    mask_subdir = os.path.join(MASK_DIR, subfolder_clean)
    os.makedirs(mask_subdir, exist_ok=True)

    if not os.path.isdir(image_subdir):
        continue

    for file_name in tqdm(os.listdir(image_subdir), desc=f"Processing {subfolder_clean}"):
        # Skip hidden files
        if file_name.startswith('.'):
            continue

        fname_no_ext = os.path.splitext(file_name)[0].lower().strip()

        # Match filename to COCO annotation
        img_id = filename_to_id.get(fname_no_ext)
        if img_id is None:
            for ext in [".jpg", ".jpeg", ".png"]:
                img_id = filename_to_id.get(fname_no_ext + ext.replace('.', '').lower())
                if img_id:
                    break

        if img_id is None:
            print(f"⚠️ No annotation for {file_name}")
            continue

        # Load image info
        img_info = coco.loadImgs(img_id)[0]
        h, w = img_info["height"], img_info["width"]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Load annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            ann_mask = coco.annToMask(ann)
            mask[ann_mask > 0] = 255

        # Save mask (overwrites if already exists)
        mask_name = fname_no_ext + ".png"
        cv2.imwrite(os.path.join(mask_subdir, mask_name), mask)

print("\n✅ Training masks created successfully in respective subfolders!")
