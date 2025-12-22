# test_predict.py
from .predict import predict_image

image_path = "/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/processed/test/images/cylindrical/16_1_2_cyl.jpg"  # Update with an actual test image path
class_id = predict_image(image_path)

classes = ["Cylindrical", "Pouch", "Prismatic"]
print("Predicted class:", classes[class_id])
