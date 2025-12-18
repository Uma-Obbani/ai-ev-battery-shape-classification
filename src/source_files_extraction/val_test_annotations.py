from pathlib import Path
import json

# -----------------------------
# Paths (update these)
# -----------------------------
RAW_JSON_FILE = Path("/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/zenodo_raw_files/recybat24_aug/recybat24-aug/val/annotations.json")
BASE_IMAGES_DIR = Path("/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/augumented")
BASE_ANN_DIR = Path("/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/augumented")

splits = [ "val", "test"]
classes = ["cylindrical", "prismatic", "pouch"]

# -----------------------------
# Load main JSON
# -----------------------------
with open(RAW_JSON_FILE) as f:
    data = json.load(f)

if isinstance(data, dict):
    if "images" in data:
        images_data = data["images"]
    else:
        raise ValueError("JSON does not contain 'images' key")
elif isinstance(data, list):
    images_data = data
else:
    raise TypeError("Unexpected JSON format")

annotations_dict = {entry["file_name"]: entry for entry in images_data}

# -----------------------------
# Loop over splits, classes, and images
# -----------------------------
for split in splits:
    for cls in classes:
        img_folder = BASE_IMAGES_DIR / split / "images" / cls
        ann_folder = BASE_ANN_DIR / split / "annotations" / cls
        ann_folder.mkdir(parents=True, exist_ok=True)

        for img_file in img_folder.glob("*.jpg"):
            fname = img_file.name
            if fname in annotations_dict:
                ann = annotations_dict[fname]
                out_file = ann_folder / (img_file.stem + ".json")
                with open(out_file, "w") as f:
                    json.dump(ann, f)
            else:
                print(f"❌ Annotation not found for image: {fname}")

print("✅ All available annotations copied for train, val, and test!")
