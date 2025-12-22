import os

# Path to your train folder
train_dir = "/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/zenodo_raw_files/recybat24_aug/recybat24-aug/train"

# List classes in the train folder
classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
print("Classes found in train folder:", classes)

# Count number of files in each class
for cls in classes:
    cls_path = os.path.join(train_dir, cls)
    files = [f for f in os.listdir(cls_path) if f.endswith(('.jpg','.png'))]
    print(f"Class '{cls}': {len(files)} images")

# Total images in train folder
total_images = sum(len(os.listdir(os.path.join(train_dir, cls))) for cls in classes)
print(f"Total images in train folder: {total_images}")
