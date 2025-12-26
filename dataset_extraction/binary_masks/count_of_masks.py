import os

def count_images_recursive(root):
    return sum(
        1
        for _, _, files in os.walk(root)
        for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

image_folders = {
    "train": "/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/processed/train/masks",
    "val":   "/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/processed/val/masks",
    "test":  "/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/processed/test/masks",
}

print("\n📊 IMAGE COUNTS (RECURSIVE)")
for split, folder in image_folders.items():
    print(f"{split}: {count_images_recursive(folder)} images")
