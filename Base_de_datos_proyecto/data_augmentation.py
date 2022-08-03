import cv2
from cv2 import normalize
import numpy as np
import torchvision.transforms as transforms

transform = transforms.Compose(
    [
        transforms.Resize(width=256, height=256),
        transforms.RandomCrop(width=1280, height=720),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomGrayscale(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        transforms.ToTensor(),
        transforms.Normalize(mean = 0, std = 1),
        
    ]
)
