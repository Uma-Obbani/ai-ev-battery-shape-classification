# src/dataset.py
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

# Transformations for ResNet
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
val_transforms = train_transforms
test_transforms = train_transforms

import os

INPUT_TYPE = os.getenv("INPUT_TYPE", "images")



# Dataset paths
#train_path = "/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/processed/train/images"
#val_path = "/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/processed/val/images"
#test_path = "/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/processed/test/images"

train_path = f"data/processed/train/{INPUT_TYPE}"
val_path = f"data/processed/val/{INPUT_TYPE}"
test_path = f"data/processed/test/{INPUT_TYPE}"

# Datasets
train_dataset = ImageFolder(train_path, transform=train_transforms)
val_dataset = ImageFolder(val_path, transform=val_transforms)
test_dataset = ImageFolder(test_path, transform=test_transforms)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)