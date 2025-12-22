import torch
from torch import nn, optim
from pathlib import Path
import os

from .model import get_model
from .dataset import train_loader, val_loader,train_dataset  #et  # Use your dataset.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model for 3 classes
num_classes = len(train_dataset.classes)
model = get_model(num_classes=3).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

best_val_loss = float("inf")

for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_loss /= len(val_loader)
    val_acc = correct / total

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs("model", exist_ok=True)
        torch.save(model.state_dict(), Path("model/best_model.pth"))
        print("Saved Best Model!")

print("Training completed!")
