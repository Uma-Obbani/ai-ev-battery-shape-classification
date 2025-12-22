import torch
from torchvision import transforms
from PIL import Image
from .model import get_model
from .dataset import train_dataset  # To get class names

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
num_classes = len(train_dataset.classes)
model = get_model(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("model/best_model.pth", map_location=device))
model.eval()

# Transform for a single image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def predict_image(image_path):
    """Predict class of a single image"""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, pred_idx = torch.max(outputs, 1)
    
    class_name = train_dataset.classes[pred_idx.item()]
    return class_name

# Example usage
if __name__ == "__main__":
    test_image_path = "/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/processed/test/images/prismatic/1_1_R_2_pri.jpg"
    predicted_class = predict_image(test_image_path)
    print(f"Predicted Class: {predicted_class}")
