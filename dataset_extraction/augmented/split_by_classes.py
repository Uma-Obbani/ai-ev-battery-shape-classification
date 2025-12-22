from pathlib import Path
import shutil

# Suffix → class folder mapping
SUFFIX_MAP = {
    "_cyl": "cylindrical",
    "_po": "pouch",
    "_pri": "prismatic",
}

def move_images(split: str):
    source_dir = Path(f"data/raw/{split}")
    dest_dir = Path(f"data/processed/{split}")

    if not source_dir.exists():
        print(f"❌ Source folder does not exist: {source_dir}")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for img_path in source_dir.glob("*.jpg"):
        filename = img_path.name.lower()

        for suffix, class_folder in SUFFIX_MAP.items():
            if filename.endswith(suffix + ".jpg"):
                target_dir = dest_dir / class_folder
                target_dir.mkdir(parents=True, exist_ok=True)

                shutil.move(img_path, target_dir / img_path.name)
                moved += 1
                break

    print(f"✅ {split}: moved {moved} images.")

if __name__ == "__main__":
    # 🚨 ONLY VAL — train already done
    move_images("val")
