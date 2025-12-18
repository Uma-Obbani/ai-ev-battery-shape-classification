import json
import numpy as np
from pathlib import Path
import cv2

BASE_DIR = Path("/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/augumented")  # Replace with your Trine folder path
SPLITS = ["train", "val", "test"]
CLASSES = ["cylindrical", "prismatic", "pouch"]

CLASS_MAP = {"cylindrical": 1, "pouch": 2, "prismatic": 3}

for split in SPLITS:
    for cls in CLASSES:
        # Define paths for images, annotations, masks
        img_folder = BASE_DIR / split / "images" / cls
        ann_folder = BASE_DIR / split / "annotations" / cls
        mask_folder = BASE_DIR / split / "masks" / cls
        mask_folder.mkdir(parents=True, exist_ok=True)

        # Loop over images
        for img_file in img_folder.glob("*.jpg"):
            fname = img_file.stem

            # Load image to get dimensions
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"❌ Cannot read image: {img_file}")
                continue
            height, width = img.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)

            # Load corresponding JSON annotation
            ann_file = ann_folder / (fname + ".json")
            if ann_file.exists():
                with open(ann_file) as f:
                    ann = json.load(f)

                # Draw polygons on mask
                segmentations = ann.get("segmentation", [])
                for seg in segmentations:
                    pts = np.array(seg).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [pts], color=CLASS_MAP[cls])
            else:
                print(f"⚠️ Annotation not found for {fname}, mask will be empty")

            # Save mask
            mask_file = mask_folder / (fname + ".png")
            cv2.imwrite(str(mask_file), mask)

print("✅ Masks generated for all splits and classes in Train!")
