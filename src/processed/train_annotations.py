from pathlib import Path
import json

# -----------------------------
# Paths (update these)
# -----------------------------
RAW_JSON_FILE = Path("/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/raw/train/annotations.json")
IMAGES_DIR = Path("/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/processed/train/images")
ANN_DIR = Path("/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/processed/train/annotations")

# Class subfolders
classes = ["cylindrical", "prismatic", "pouch"]

# -----------------------------
# Load main JSON
# -----------------------------
with open(RAW_JSON_FILE) as f:
    data = json.load(f)

# Check JSON structure
if isinstance(data, dict):
    # Assume COCO-style: images are under 'images' key
    if "images" in data:
        images_data = data["images"]
    else:
        raise ValueError("JSON does not contain 'images' key")
elif isinstance(data, list):
    images_data = data
else:
    raise TypeError("Unexpected JSON format")

# Create a dictionary for fast lookup
annotations_dict = {entry["file_name"]: entry for entry in images_data}

# -----------------------------
# Loop over classes and images
# -----------------------------
for cls in classes:
    img_folder = IMAGES_DIR / cls
    ann_folder = ANN_DIR / cls
    ann_folder.mkdir(parents=True, exist_ok=True)

    for img_file in img_folder.glob("*.jpg"):
        fname = img_file.name
        if fname in annotations_dict:
            ann = annotations_dict[fname]
            # Save individual JSON for the image
            out_file = ann_folder / (img_file.stem + ".json")
            with open(out_file, "w") as f:
                json.dump(ann, f)
        else:
            print(f"❌ Annotation not found for image: {fname}")

print("✅ All available annotations copied successfully!")
