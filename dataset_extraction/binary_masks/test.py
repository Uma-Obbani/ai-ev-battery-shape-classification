import os

def count_masks(root):
    return sum(
        1 for _, _, files in os.walk(root)
        for f in files if f.endswith(".png")
    )
    
    
VAL_MASK_DIR ="/Users/umaobbani/Documents/DSR_ai-ev-battery-shape-classification_portfolio/ai-ev-battery-shape-classification/data/processed/val/masks"

    

print("FINAL CHECK")
print("val masks:", count_masks(VAL_MASK_DIR))
