from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms
from PIL import Image
import io
from pathlib import Path

from src.model import get_model
from src.dataset import train_dataset

# -------------------------------------------------
# App & Device
# -------------------------------------------------
app = FastAPI(title="EV Battery Shape Classification API")

device = "mps" if torch.backends.mps.is_available() else "cpu"

# -------------------------------------------------
# Class info
# -------------------------------------------------
class_names = train_dataset.classes
num_classes = len(class_names)

# -------------------------------------------------
# Load IMAGE model (once)
# -------------------------------------------------
image_model = get_model(num_classes=num_classes).to(device)
image_model.load_state_dict(
    torch.load(Path("model/best_model_images.pth"), map_location=device)
)
image_model.eval()

# -------------------------------------------------
# Load CUTOUT model (once)
# -------------------------------------------------
cutout_model = get_model(num_classes=num_classes).to(device)
cutout_model.load_state_dict(
    torch.load(Path("model/best_model_cutouts.pth"), map_location=device)
)
cutout_model.eval()

# -------------------------------------------------
# Image transforms
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------
# Shared inference function
# -------------------------------------------------
def run_inference(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return {
        "predicted_class": class_names[pred.item()],
        "confidence": round(conf.item(), 4),
        "probabilities": {
            class_names[i]: round(probs[0][i].item(), 4)
            for i in range(len(class_names))
        }
    }

# -------------------------------------------------
# Endpoint: IMAGE model only
# -------------------------------------------------
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    result = run_inference(image_model, image)
    result["model_used"] = "image"

    return result

# -------------------------------------------------
# Endpoint: CUTOUT model only
# -------------------------------------------------
@app.post("/predict/cutout")
async def predict_cutout(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    result = run_inference(cutout_model, image)
    result["model_used"] = "cutout"

    return result

# -------------------------------------------------
# Endpoint: BOTH models (comparison)
# -------------------------------------------------
@app.post("/predict/both")
async def predict_both(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    image_result = run_inference(image_model, image)
    cutout_result = run_inference(cutout_model, image)

    note = (
        "Models agree"
        if image_result["predicted_class"] == cutout_result["predicted_class"]
        else "Models disagree – possible background bias"
    )

    return {
        "image_model": image_result,
        "cutout_model": cutout_result,
        "note": note
    }
