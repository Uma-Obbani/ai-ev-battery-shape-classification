import torch
from pathlib import Path
from .model import get_model
from .dataset import test_loader, train_dataset  # Use train_dataset for class info if needed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
num_classes = len(train_dataset.classes)  # Get number of classes automatically
model = get_model(num_classes=num_classes).to(device)

# Load trained model
model_path = Path("model") / "best_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Evaluate on test set
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
