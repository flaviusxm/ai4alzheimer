import torch
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from cnn import model, preprocess
import torch.nn.functional as F

# --- Configuration ---
CHECKPOINT_PATH = "checkpoints/best_model.pth"  # Default local path
if os.path.exists("/content/drive/MyDrive/alzheimer_checkpoints/best_model.pth"):
    CHECKPOINT_PATH = "/content/drive/MyDrive/alzheimer_checkpoints/best_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS_MAP = {0: "mild", 1: "moderate", 2: "non", 3: "very-mild"}
NUM_SAMPLES = 6  # Number of random images to show

def load_model():
    print(f"Loading model from {CHECKPOINT_PATH}...")
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Please ensure you have trained the model or the path is correct.")
        exit()
    model.to(DEVICE)
    model.eval()
    return model

def get_random_images(root_dirs=None):
    if root_dirs is None:
        root_dirs = ["mild-dementia", "moderate-dementia", "non-dementia", "very-mild-dementia"]
    
    images = []
    
    # Check if directories exist
    valid_dirs = [d for d in root_dirs if os.path.exists(d)]
    if not valid_dirs:
        print("Error: Dataset directories not found in current folder.")
        print("Please run this script where the dataset folders (e.g., 'mild-dementia') are located.")
        return []

    for _ in range(NUM_SAMPLES):
        # Pick a random class
        class_dir = random.choice(valid_dirs)
        clean_class_name = class_dir.replace("-dementia", "") # e.g. mild
        
        # Pick a random file
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files: 
            continue
            
        filename = random.choice(files)
        filepath = os.path.join(class_dir, filename)
        
        images.append((filepath, clean_class_name))
        
    return images

def visualize_inferences():
    model = load_model()
    samples = get_random_images()
    
    if not samples:
        return

    plt.figure(figsize=(15, 10))
    
    print("\n--- Running Inference ---")
    
    for i, (filepath, true_label) in enumerate(samples):

        original_img = Image.open(filepath).convert("RGB")
        input_tensor = preprocess(original_img).unsqueeze(0).to(DEVICE)
   
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)
            
        predicted_label = LABELS_MAP[predicted_idx.item()]
        conf_score = confidence.item() * 100
        
        color = 'green' if predicted_label == true_label else 'red'
        
        print(f"[{i+1}] True: {true_label:<10} | Pred: {predicted_label:<10} | Conf: {conf_score:.2f}% | File: {os.path.basename(filepath)}")
        
        # Plot
        ax = plt.subplot(2, 3, i + 1)
        ax.imshow(original_img)
        ax.set_title(f"True: {true_label}\nPred: {predicted_label} ({conf_score:.1f}%)", color=color, fontweight='bold')
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()
    print("\nInference complete. Check the plot window.")

if __name__ == "__main__":
    visualize_inferences()
