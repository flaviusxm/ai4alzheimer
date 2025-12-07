import torch
import torch.nn as nn
from torchvision import models, transforms

weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
preprocess = weights.transforms()

model = models.efficientnet_v2_s(weights=weights)
for param in model.parameters():  # freeze parameters in the base model
    param.requires_grad = False
