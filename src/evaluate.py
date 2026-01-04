print("🔥🔥🔥 THIS IS THE EVALUATE.PY FILE I AM EDITING")
print("FILE PATH:", __file__)

import torch
from torch import nn
from pathlib import Path
import os
from sklearn.metrics import confusion_matrix, classification_report

from .model import get_model
from .dataset import INPUT_TYPE, val_loader, train_dataset
from utils.logger import get_logger

# -------------------------------------------------
# Project root & logger
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

logger = get_logger(
    "evaluation",
    os.path.join(PROJECT_ROOT, "training_logs", "eval.log")
)

device = "mps" if torch.backends.mps.is_available() else "cpu"


def evaluate():
    logger.info(f"Evaluation started | INPUT_TYPE={INPUT_TYPE}")

    num_classes = len(train_dataset.classes)

    model = get_model(num_classes=num_classes).to(device)

    model_path = Path(PROJECT_ROOT) / "model" / f"best_model_{INPUT_TYPE}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    criterion = nn.CrossEntropyLoss()

    val_loss = 0.0
    correct = 0
    total = 0

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
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = correct / total

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=train_dataset.classes
    )

    logger.info(f"Validation Loss: {val_loss:.4f}")
    logger.info(f"Validation Accuracy: {val_acc:.4f}")
    logger.info("Confusion Matrix:\n%s", cm)
    logger.info("Classification Report:\n%s", report)

    print(f"Validation Accuracy: {val_acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)


if __name__ == "__main__":
    evaluate()
