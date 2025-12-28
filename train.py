import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from dataset import AlzheimerDataset
from cnn import model, preprocess, set_trainable_blocks
from utilities import FocalLoss
from global_constants import IMAGE_SIZE
#Config
BATCH_SIZE = 32
LEARNING_RATE_WARMUP = 1e-3
LEARNING_RATE_FINETUNE = 1e-4 
WARMUP_EPOCHS = 2
FINETUNE_EPOCHS = 10
TOTAL_EPOCHS = WARMUP_EPOCHS + FINETUNE_EPOCHS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Checkpoint Handler
if os.path.exists("/content/drive/MyDrive"):
    CHECKPOINT_DIR = "/content/drive/MyDrive/alzheimer_checkpoints"
    print(f"Google Drive detected. Saving checkpoints to: {CHECKPOINT_DIR}")
else:
    CHECKPOINT_DIR = "checkpoints"
    print(f"Saving checkpoints locally to: {CHECKPOINT_DIR}")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


LABELS_MAP = {0: "mild", 1: "moderate", 2: "non", 3: "very-mild"}
all_image_paths = []
all_labels = []

print("Scanning directories...")
for label_int, label_str in LABELS_MAP.items():
    folder_name = f"{label_str}-dementia"
    if not os.path.exists(folder_name):
        print(f"Warning: Directory {folder_name} not found. Skipping.")
        continue
        
    files = os.listdir(folder_name)
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_image_paths.append(os.path.join(folder_name, f))
            all_labels.append(label_int)

print(f"Total images found: {len(all_image_paths)}")

#Split
X_train, X_temp, y_train, y_temp = train_test_split(
    all_image_paths, all_labels, test_size=0.3, stratify=all_labels, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

#Augmentation & Transforms
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#Datasets & DataLoaders
train_dataset = AlzheimerDataset(X_train, y_train, transform=train_transforms)
val_dataset = AlzheimerDataset(X_val, y_val, transform=val_transforms)
test_dataset = AlzheimerDataset(X_test, y_test, transform=val_transforms)

#Sample Weights
class_counts = np.bincount(y_train)
class_weights = 1. / class_counts
sample_weights = [class_weights[label] for label in y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

#Setup
model = model.to(DEVICE)

criterion = FocalLoss(gamma=2.0) 

def train_one_epoch(epoch_index, optimizer, scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch_index} [Train]")
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        loop.set_postfix(loss=loss.item(), acc=100 * correct / total)
    
    if scheduler:
        scheduler.step()
        
    return running_loss / len(train_loader), 100 * correct / total

def validate(epoch_index):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / val_total
    print(f"Epoch {epoch_index} Validation: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
    return val_loss, val_acc

#Start Training
best_val_loss = float('inf')

print("\n=== STAGE 1: WARMUP (Training Classifier Only) ===")
set_trainable_blocks(0) 
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_WARMUP, weight_decay=1e-4)

for epoch in range(1, WARMUP_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(epoch, optimizer)
    val_loss, val_acc = validate(epoch)

print("\n=== STAGE 2: FINE-TUNING (Last 2 Blocks + Classifier) ===")
set_trainable_blocks(2) 
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_FINETUNE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)

for epoch in range(WARMUP_EPOCHS + 1, TOTAL_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(epoch, optimizer, scheduler)
    val_loss, val_acc = validate(epoch)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
        print(">>> Best Model Saved!")

print("Training complete.")
