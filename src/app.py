from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms
from pathlib import Path
from .model import get_model
from .dataset import train_dataset  # For class names

app = FastAPI(title="EV Battery Shape Prediction")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
num_classes = len(train_dataset.classes)
model = get_model(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("model/best_model.pth", map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

def predict_image(image: Image.Image):
    """Predict class from PIL image"""
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, pred_idx = torch.max(outputs, 1)
    return train_dataset.classes[pred_idx.item()]

# API endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image = Image.open(file.file)
        predicted_class = predict_image(image)
        return JSONResponse(content={"predicted_class": predicted_class})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
