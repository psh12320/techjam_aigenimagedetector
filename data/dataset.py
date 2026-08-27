import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from pathlib import Path
import random

class AIGCDataset(Dataset):
    """Dataset for AI-generated vs real images"""

    def __init__(self, root, split='train', transform=None, train_ratio=0.8, val_ratio=0.1, seed=42):
        """
        Args:
            root: Root directory with 'real' and 'fake' subdirectories
            split: 'train', 'val', or 'test'
            transform: Transform to apply to images
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            seed: Random seed for split reproducibility
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # Load all image paths
        real_images = list((self.root / 'real').glob('*.[jp][pn]g'))
        fake_images = list((self.root / 'fake').glob('*.[jp][pn]g'))

        # Create (path, label) pairs
        self.samples = [(img, 0) for img in real_images] + [(img, 1) for img in fake_images]

        # Split dataset
        random.seed(seed)
        random.shuffle(self.samples)

        n_total = len(self.samples)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        if split == 'train':
            self.samples = self.samples[:n_train]
        elif split == 'val':
            self.samples = self.samples[n_train:n_train + n_val]
        elif split == 'test':
            self.samples = self.samples[n_train + n_val:]
        else:
            raise ValueError(f"Invalid split: {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label
