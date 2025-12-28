import os
import cv2 as cv
from PIL import Image
import torch
from torch.utils.data import Dataset
from global_constants import IMAGE_SIZE, SCALED_DIMENSIONS, PAD_AMOUNT

class AlzheimerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = cv.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image at {img_path}")

        h_old, w_old = img.shape[0], img.shape[1]
        

        if h_old not in SCALED_DIMENSIONS:
             img = cv.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv.INTER_CUBIC)
        else:
            h_scaled, w_scaled = SCALED_DIMENSIONS[h_old]
            img = cv.resize(img, (w_scaled, h_scaled), interpolation=cv.INTER_CUBIC)

            if h_scaled != IMAGE_SIZE:
                pad_top, pad_bottom = PAD_AMOUNT[h_old]
                img = cv.copyMakeBorder(img, top=pad_top, bottom=pad_bottom, left=0, right=0,
                                        borderType=cv.BORDER_CONSTANT, value=img[0, 0].tolist())
            elif w_scaled != IMAGE_SIZE:
                pad_left, pad_right = PAD_AMOUNT[h_old]
                img = cv.copyMakeBorder(img, top=0, bottom=0, left=pad_left, right=pad_right,
                                        borderType=cv.BORDER_CONSTANT, value=img[0, 0].tolist())

        
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

 
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        
        return img, label
