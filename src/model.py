# src/model.py
from torchvision.models import resnet18
from torch import nn

def get_model(num_classes=3):
    """ResNet-18 for 3 classes: Cylindrical, Pouch, Prismatic"""
    model = resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )
    return model
