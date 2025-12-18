import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(
    '/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/augmented/train/masks/cylindrical/1_1_1_cyl_rotated_15_bright_0.5.png',
    cv2.IMREAD_GRAYSCALE
)

print("Mask shape:", img.shape)



# =========================
# Step 1: Load the image or array
# =========================

# Option 1: Load an image from file (grayscale)
# Replace 'image.png' with your file path


# Option 2: If you already have a NumPy array, uncomment this
# img = your_array_here

# =========================
# Step 2: Check basic info
# =========================
print("Shape:", img.shape)
print("Data type:", img.dtype)
print("Original unique values:", np.unique(img))
print("Value range: {} to {}".format(np.min(img), np.max(img)))

# =========================
# Step 3: Check if array is effectively empty
# =========================
if np.max(img) == 0:
    print("Warning: array contains only zeros!")

# =========================
# Step 4: Optional scaling for visualization
# If the array is float and values are tiny, scale to 0-255
# =========================
if np.issubdtype(img.dtype, np.floating):
    img_scaled = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8) * 255
    img_scaled = img_scaled.astype(np.uint8)
else:
    img_scaled = img

print("Mapped unique values:", np.unique(img_scaled))

# =========================
# Step 5: Visualize
# =========================
plt.figure(figsize=(6,6))
plt.imshow(img_scaled, cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.title("Array Visualization")
plt.show()
