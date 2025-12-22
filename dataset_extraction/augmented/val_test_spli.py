from pathlib import Path
import shutil
import random
import json

# -------------------------------
# Configuration
# -------------------------------
random.seed(42)

VAL_DIR = Path("data/processed/val")      # current validation folder (already has class folders)
TEST_DIR = Path("data/processed/test")    # folder for test split
ANNOTATIONS_FILE = Path("data/raw/val/annotations.json")  # original COCO JSON
TEST_FRACTION = 0.5  # 50-50 split

# Output JSON files
VAL_JSON_OUT = VAL_DIR / "val_annotations.json"
TEST_JSON_OUT = TEST_DIR / "test_annotations.json"

# -------------------------------
# Step 1: Load COCO annotations
# -------------------------------
with open(ANNOTATIONS_FILE) as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

# Map file_name -> image_id for quick lookup
filename_to_id = {img["file_name"]: img["id"] for img in images}

# -------------------------------
# Step 2: Prepare class folders
# -------------------------------
# Dictionary: class_name -> list of image paths
split_info = {}
for class_folder in VAL_DIR.iterdir():
    if class_folder.is_dir():
        split_info[class_folder.name] = list(class_folder.glob("*.jpg"))

# Ensure test class folders exist
for class_name in split_info.keys():
    (TEST_DIR / class_name).mkdir(parents=True, exist_ok=True)

# -------------------------------
# Step 3: Move 50% images per class to test
# -------------------------------
test_image_filenames = set()
for class_name, img_list in split_info.items():
    random.shuffle(img_list)
    n_test = int(len(img_list) * TEST_FRACTION)
    test_images = img_list[:n_test]

    for img_path in test_images:
        shutil.move(img_path, TEST_DIR / class_name / img_path.name)
        test_image_filenames.add(img_path.name)

print(f"✅ Images moved to test folder. Test images count: {len(test_image_filenames)}")

# -------------------------------
# Step 4: Split annotations based on moved images
# -------------------------------
val_annotations = []
test_annotations = []

for ann in annotations:
    img_id = ann["image_id"]
    # Find corresponding filename
    file_name = next((fname for fname, id_ in filename_to_id.items() if id_ == img_id), None)
    if file_name is None:
        continue

    if file_name in test_image_filenames:
        test_annotations.append(ann)
    else:
        val_annotations.append(ann)

# Split images for JSON
val_images = [img for img in images if img["file_name"] not in test_image_filenames]
test_images = [img for img in images if img["file_name"] in test_image_filenames]

# -------------------------------
# Step 5: Save new JSON files
# -------------------------------
VAL_JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
TEST_JSON_OUT.parent.mkdir(parents=True, exist_ok=True)

with open(VAL_JSON_OUT, "w") as f:
    json.dump({"images": val_images, "annotations": val_annotations, "categories": categories}, f, indent=4)

with open(TEST_JSON_OUT, "w") as f:
    json.dump({"images": test_images, "annotations": test_annotations, "categories": categories}, f, indent=4)

print(f"✅ Annotations split complete: {len(val_images)} val images, {len(test_images)} test images")
