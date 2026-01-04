import torch
from pathlib import Path
import os
from PIL import Image

from .model import get_model
from .dataset import test_transforms, train_dataset, INPUT_TYPE
from utils.logger import get_logger

# -------------------------------------------------
# Project root & logger
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

logger = get_logger(
    "testing",
    os.path.join(PROJECT_ROOT, "training_logs", "test.log")
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_image(image_path):
    num_classes = len(train_dataset.classes)

    model_path = Path(PROJECT_ROOT) / "model" / f"best_model_{INPUT_TYPE}.pth"

    model = get_model(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = test_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    class_name = train_dataset.classes[pred.item()]

    logger.info(
        f"Image: {image_path} | "
        f"Predicted: {class_name} | "
        f"Confidence: {conf.item():.4f}"
    )

    print(f"Predicted class: {class_name}")
    print(f"Confidence: {conf.item():.4f}")


if __name__ == "__main__":
    image_path = input("Enter image path: ")
    predict_image(image_path)
