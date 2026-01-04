import wandb
print("✅ TRAINING LOOP ENTERED")

import torch
from torch import nn, optim
from pathlib import Path
import os
from sklearn.metrics import confusion_matrix

from .model import get_model
from .dataset import INPUT_TYPE, train_loader, val_loader, train_dataset
from utils.logger import get_logger

# -------------------------------------------------
# Project root & logger (UNCHANGED)
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

logger = get_logger(
    "training",
    os.path.join(PROJECT_ROOT, "training_logs", "train.log")
)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "mps" if torch.backends.mps.is_available() else "cpu"

logger.info("Training started")

wandb.init(
    project="AI_EV_BATTERY_IMAGE_CLASSIFICATION",
    name=f"resnet18-base-{INPUT_TYPE}",
    config={
        "model": "resnet18",
        "dataset": INPUT_TYPE,
        "epochs": 5,
        "optimizer": "Adam",
        "learning_rate": 0.001
    }
)


# -------------------------------------------------
# Model
# -------------------------------------------------
num_classes = len(train_dataset.classes)
model = get_model(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

best_val_loss = float("inf")

# -------------------------------------------------
# Training loop
# -------------------------------------------------
for epoch in range(epochs):

    # =========================
    # Training
    # =========================
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    # =========================
    # Validation
    # =========================
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    # =========================
    # Confusion Matrix
    # =========================
    cm = confusion_matrix(all_labels, all_preds)

    # =========================
    # Logging
    # =========================
    logger.info(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    logger.info("Confusion Matrix:\n%s", cm)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )
    wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "confusion_matrix": wandb.plot.confusion_matrix(
                y_true=all_labels,
                preds=all_preds,
                class_names=train_dataset.classes
            )
        })

    # =========================
    # Save best model
    # =========================
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs("model", exist_ok=True)
        model_path = Path(f"model/best_model_{INPUT_TYPE}.pth")
        torch.save(model.state_dict(), model_path)


        logger.info("Saved Best Model")
        print("Saved Best Model!")

logger.info("Training completed")


print("Training completed!")
wandb.finish()
