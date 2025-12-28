import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from project
from dataset import AlzheimerDataset
from cnn import model, preprocess
from global_constants import IMAGE_SIZE

# --- Configuration ---
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/best_model.pth"
if os.path.exists("/content/drive/MyDrive/alzheimer_checkpoints/best_model.pth"):
    CHECKPOINT_PATH = "/content/drive/MyDrive/alzheimer_checkpoints/best_model.pth"

LABELS_MAP = {0: "mild", 1: "moderate", 2: "non", 3: "very-mild"}

def evaluate_model():
    print(f"--- Starting Evaluation using {DEVICE} ---")
    
    # 1. Load Data (Replicating logic from train.py to ensure we use the TEST set)
    print("Reconstructing Test Set...")
    all_image_paths = []
    all_labels = []
    
    for label_int, label_str in LABELS_MAP.items():
        folder_name = f"{label_str}-dementia"
        if not os.path.exists(folder_name):
            continue
        files = os.listdir(folder_name)
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_image_paths.append(os.path.join(folder_name, f))
                all_labels.append(label_int)
    
    if not all_image_paths:
        print("Error: No images found. Run this script from the project root.")
        return

    # Stratified Split (Must match train.py random_state=42)
    _, X_temp, _, y_temp = train_test_split(
        all_image_paths, all_labels, test_size=0.3, stratify=all_labels, random_state=42
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    print(f"Test Set Size: {len(X_test)} images")

    # 2. Setup DataLoader
    test_dataset = AlzheimerDataset(X_test, y_test, transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 3. Load Model
    print(f"Loading weights from: {CHECKPOINT_PATH}")
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print("Checkpoint not found. Cannot evaluate.")
        return
        
    model.to(DEVICE)
    model.eval()

    # 4. Prediction Loop
    all_preds = []
    all_targets = []
    
    print("Running predictions...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())

    # 5. Metrics
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    
    target_names = [LABELS_MAP[i] for i in range(len(LABELS_MAP))]
    print(classification_report(all_targets, all_preds, target_names=target_names))
    
    # 6. Confusion Matrix Plot
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - EfficientNetV2-S')
    
    save_path = "confusion_matrix.png"
    plt.savefig(save_path)
    print(f"\nConfusion Matrix saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    evaluate_model()
