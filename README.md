Battery Cell Classification with ResNet-18 & FastAPI
Complete end-to-end project for classifying lithium-ion battery cells (Cylindrical, Pouch, Prismatic) using
ResNet-18 and deploying with FastAPI.
Dataset: RecyBat24
📋 Table of Contents
Project Structure
Prerequisites
Step 1: Setup Environment
Step 2: Prepare Dataset
Step 3: Train Model
Step 4: Deploy with FastAPI
Step 5: Test API
Docker Deployment
Portfolio Tips
📁 Project Structure
battery-classification/
├── data/
│ ├── train/
│ │ └── images/
│ │ ├── cylindrical/
│ │ ├── pouch/
│ │ └── prismatic/
│ ├── val/
│ │ └── images/
│ │ ├── cylindrical/
│ │ ├── pouch/
│ │ └── prismatic/
│ └── test/
│ └── images/
│ ├── cylindrical/
│ ├── pouch/
│ └── prismatic/
├── models/
│ ├── resnet18_battery_best.pth
│ ├── class_names.json
│ └── training_history.png
├── train.py
├── app.py
├── test_api.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
✅ Prerequisites
Python 3.8+
CUDA-capable GPU (optional but recommended)
8GB+ RAM
Basic knowledge of PyTorch and FastAPI
🚀 Step 1: Setup Environment
1.1 Create Virtual Environment
bash
# Create project directory
mkdir battery-classification
cd battery-classification
# Create virtual environment
python -m venv venv
# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
1.2 Install Dependencies
bash
pip install -r requirements.txt
📦 Step 2: Prepare Dataset
2.1 Download RecyBat24 Dataset
1. Visit https://zenodo.org/records/15490523
2. Download recybat24.tar.gz (262.8 MB) or recybat24
_
aug.tar.gz (2.6 GB with augmentations)
3. Extract the archive
2.2 Organize Your Data
Your data should already be organized as:
data/
├── train/images/
│ ├── cylindrical/
│ ├── pouch/
│ └── prismatic/
├── val/images/
│ ├── cylindrical/
│ ├── pouch/
│ └── prismatic/
└── test/images/
├── cylindrical/
├── pouch/
└── prismatic/
Important: Ensure your folder structure exactly matches this pattern. The ImageFolder loader expects this
format.
2.3 Verify Data
python
import os
# Quick verification script
data_splits = ['train', 'val', 'test']
classes = ['cylindrical', 'pouch', 'prismatic']
for split in data_splits:
print(f"\n{split.upper()}:")
for cls in classes:
path = f"./data/{split}/images/{cls}"
if os.path.exists(path):
count = len(os.listdir(path))
print(f" {cls}: {count} images")
else:
print(f" {cls}: MISSING!")
🎯 Step 3: Train Model
3.1 Configure Training Parameters
Edit train.py if needed to adjust:
batch
_
size : Default 32 (reduce if GPU memory issues)
num
_
epochs : Default 50
learning_
rate : Default 0.001
img_
size : Default 224
3.2 Start Training
bash
python train.py
What happens during training:
ResNet-18 loads with ImageNet pretrained weights
Only the final classification layer is modified for 3 classes
Data augmentation applied to training set (rotation, flip, color jitter)
Training with Adam optimizer and ReduceLROnPlateau scheduler
Best model saved based on validation accuracy
Training history plot generated
Expected Output:
Loading datasets...
Classes: ['cylindrical', 'pouch', 'prismatic']
Train samples: XXXX
Val samples: XXXX
Test samples: XXXX
Model created and moved to cuda
Epoch 1/50
Training: 100%|████████| ...
Train Loss: 0.8234, Train Acc: 68.45%
Val Loss: 0.5123, Val Acc: 82.30%
✓ Best model saved with validation accuracy: 82.30%
...
3.3 Monitor Training
The script will:
Show progress bars for each epoch
Display train/val loss and accuracy
Save best model automatically
Generate training curves at the end
Training typically takes:
CPU only: 2-4 hours
GPU (CUDA): 15-30 minutes
3.4 Output Files
After training completes:
models/
├── resnet18_battery_best.pth # Best model (use this!)
├── resnet18_battery_classifier.pth # Final model
├── class_names.json # Class mappings
└── training_history.png # Loss/accuracy plots
🌐 Step 4: Deploy with FastAPI
4.1 Verify Model Files
Ensure these files exist:
models/resnet18
_
battery_
best.pth
models/class
_
names.json
4.2 Start FastAPI Server
bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
Server will start at: http://localhost:8000
4.3 Check API Documentation
Open your browser and visit:
Interactive docs: http://localhost:8000/docs
Alternative docs: http://localhost:8000/redoc
Root endpoint: http://localhost:8000
🧪 Step 5: Test API
5.1 Test Health Endpoint
bash
curl http://localhost:8000/health
5.2 Test with Python Client
bash
python test_api.py
Update the image paths in test
_
api.py to point to your actual test images.
5.3 Test with cURL
Single Image:
bash
curl -X POST "http://localhost:8000/predict" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-F "file=@/path/to/battery_image.jpg"
Expected Response:
json
{
"success": true,
"filename": "battery_image.jpg",
"prediction": {
"predicted_class": "cylindrical",
"confidence": 0.9534,
"all_probabilities": {
"cylindrical": 0.9534,
"pouch": 0.0312,
"prismatic": 0.0154
}
}
}
5.4 Test with Postman
1. Create new POST request to http://localhost:8000/predict
2. Select Body → form-data
3. Add key file (type: File)
4. Upload battery image
5. Send request
🐳 Docker Deployment
6.1 Create .dockerignore
Create .dockerignore :
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.git/
data/
*.png
.env
.vscode/
6.2 Build Docker Image
bash
docker build -t battery-classifier:latest .
6.3 Run Container
bash
docker run -d -p 8000:8000 --name battery-api battery-classifier:latest
6.4 Test Docker Deployment
bash
curl http://localhost:8000/health
6.5 Stop Container
bash
docker stop battery-api
docker rm battery-api
📊 Model Performance Evaluation
Create Evaluation Script
python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Load test data
test_transforms = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder('./data/test/images', transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# Load model
checkpoint = torch.load('./models/resnet18_battery_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
# Get predictions
all_preds = []
all_labels = []
with torch.no_grad():
for inputs, labels in test_loader:
outputs = model(inputs)
_, preds = torch.max(outputs, 1)
all_preds.extend(preds.cpu().numpy())
all_labels.extend(labels.cpu().numpy())
# Classification report
class_names = test_dataset.classes
print(classification_report(all_labels, all_preds, target_names=class_names))
# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('./models/confusion_matrix.png')
🎨 Portfolio Tips
1. GitHub Repository Structure
Create a professional README with:
Project overview and motivation
Dataset description with citation
Architecture diagram
Training results (accuracy, loss curves)
API endpoints documentation
Sample predictions with images
2. Key Metrics to Highlight
Model Size: ~44.7M parameters
Training Time: X minutes on GPU
Test Accuracy: X%
Inference Time: ~X ms per image
API Response Time: ~X ms
3. Visualization Ideas
Training/validation curves
Confusion matrix
Sample predictions (correct and incorrect)
Grad-CAM visualizations (what the model focuses on)
API response time benchmarks
4. Demo Deployment
Deploy to:
Hugging Face Spaces (free, easy)
Railway (free tier available)
Render (free tier available)
AWS EC2 (more professional)
5. Additional Features to Add
Batch processing endpoint
Model versioning
Logging and monitoring
Rate limiting
API authentication
Confidence threshold filtering
Image preprocessing validation
6. Documentation Enhancements
Add OpenAPI schema customization
Include example requests/responses
Create Jupyter notebook for exploration
Add model architecture diagram
Document hyperparameter tuning process
🔧 Troubleshooting
Issue: CUDA Out of Memory
Solution: Reduce batch size in train.py
python
batch_size = 16 # or even 8
Issue: API Model Not Found
Solution: Check model paths match
bash
ls -la models/
# Should see: resnet18_battery_best.pth and class_names.json
Issue: Low Accuracy
Potential causes:
Insufficient training epochs
Learning rate too high/low
Data imbalance between classes
Images not properly preprocessed
Solutions:
Increase epochs to 100
Try different learning rates (0.0001, 0.01)
Check class distribution
Verify image quality and labels
Issue: Slow Inference
Solutions:
Enable GPU inference
Use model quantization
Batch multiple requests
Use ONNX runtime
📚 Additional Resources
PyTorch Documentation: https://pytorch.org/docs/
FastAPI Tutorial: https://fastapi.tiangolo.com/tutorial/
ResNet Paper: https://arxiv.org/abs/1512.03385
RecyBat24 Paper: DOI: 10.1038/s41597-025-05211-5
🎓 Next Steps
1. Improve Model:
Try ResNet-50 or EfficientNet
Implement test-time augmentation
Use ensemble methods
2. Enhance API:
Add batch processing
Implement caching
Add API key authentication
Create rate limiting
3. Deploy to Production:
Set up CI/CD pipeline
Add monitoring (Prometheus/Grafana)
Implement logging (ELK stack)
Use load balancer
4. Create Web Interface:
Build React/Vue frontend
Add drag-and-drop upload
Display prediction visualization
Show confidence scores
📝 Citation
If you use this project or the RecyBat24 dataset:
bibtex
@dataset{recybat24,
author = {Acaro Chacón, Ximena Carolina and
Lo Scudo, Fabrizio and
Cappuccino, Gregorio and
Dodaro, Carmine},
title = {RecyBat24: a dataset for detecting lithium-ion
batteries in electronic waste disposal},
year = 2025,
publisher = {Zenodo},
doi = {10.1038/s41597-025-05211-5}
}
📄 License
This project template is provided as-is for educational and portfolio purposes.
The RecyBat24 dataset is licensed under CC BY-NC-ND 4.0.
🤝 Contributing
This is a portfolio project, but suggestions are welcome! Feel free to:
Open issues for bugs
Suggest improvements
Share your results
Happy Coding! 🚀