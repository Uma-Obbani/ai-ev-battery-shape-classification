import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

# ---------------- CONFIG ----------------
VAL_JSON = "/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/raw/val/instance_segmentation_annotations.json"

VAL_IMAGE_DIR = "/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/processed/val/images"
TEST_IMAGE_DIR = "/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/processed/test/images"

VAL_MASK_DIR = "/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/processed/val/masks"
TEST_MASK_DIR = "/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/processed/test/masks"

# ---------------- FUNCTIONS ----------------
def generate_masks(coco_json, image_dir, mask_dir):
    """
    Generate masks strictly for images present in the folder.
    Counts masks for all categories automatically.
    """
    coco = COCO(coco_json)
    os.makedirs(mask_dir, exist_ok=True)

    # List all image filenames in the folder
    folder_filenames = set(os.listdir(image_dir))

    # Map filenames to image IDs (only those present in folder)
    filename_to_id = {img["file_name"]: img["id"] for img in coco.dataset["images"] if img["file_name"] in folder_filenames}

    # Map category ID to name
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco.dataset['categories']}

    # Initialize counts dictionary for all categories
    counts = {cat_name: 0 for cat_name in cat_id_to_name.values()}

    # Process each image in the folder
    for file_name in tqdm(folder_filenames, desc=f"Processing {os.path.basename(image_dir)}"):
        img_id = filename_to_id.get(file_name)
        if img_id is None:
            continue  # Skip images not present in JSON

        img_info = coco.loadImgs(img_id)[0]
        h, w = img_info["height"], img_info["width"]
        mask = np.zeros((h, w), dtype=np.uint8)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            ann_mask = coco.annToMask(ann)
            mask[ann_mask > 0] = 255

            # Count mask for the category
            cat_name = cat_id_to_name.get(ann['category_id'])
            if cat_name:
                counts[cat_name] += 1

        # Save mask (overwrite if exists)
        mask_path = os.path.join(mask_dir, file_name)
        cv2.imwrite(mask_path, mask)

    # Print counts
    print(f"\n✅ Mask counts for {os.path.basename(image_dir)}:")
    for cat, c in counts.items():
        print(f"  {cat}: {c} masks")

    return counts

# ---------------- RUN ----------------
val_counts = generate_masks(VAL_JSON, VAL_IMAGE_DIR, VAL_MASK_DIR)
test_counts = generate_masks(VAL_JSON, TEST_IMAGE_DIR, TEST_MASK_DIR)

# ---------------- SUMMARY ----------------
print("\n📊 Summary of masks:")
all_categories = set(list(val_counts.keys()) + list(test_counts.keys()))
for cat in all_categories:
    print(f"{cat}: validation = {val_counts.get(cat,0)}, test = {test_counts.get(cat,0)}")
