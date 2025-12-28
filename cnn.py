import torch
import torch.nn as nn
from torchvision import models, transforms

weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
preprocess = weights.transforms()

model = models.efficientnet_v2_s(weights=weights)


# Freeze parameters in the base model initially
for param in model.parameters():
    param.requires_grad = False

def set_trainable_blocks(num_blocks=2):
    """
    Unfreezes the last `num_blocks` blocks of the features.
    If num_blocks is 0, only the classifier is trainable.
    """
    # First, freeze everything again to be sure
    for param in model.features.parameters():
        param.requires_grad = False
        
    if num_blocks > 0:
        # Unfreeze the last n blocks
        for param in model.features[-num_blocks:].parameters():
            param.requires_grad = True
            
    print(f"Model update: Last {num_blocks} feature blocks are now TRAINABLE.")

# Default: 2 blocks (as before)
set_trainable_blocks(2)

model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)

