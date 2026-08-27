import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def extract_dct_features(images):
    """Extract DCT features from images

    Args:
        images: Tensor of shape (B, 3, H, W) in range [0, 1] (normalized)

    Returns:
        features: Tensor of shape (B, 64) with DCT coefficients
    """
    B, C, H, W = images.shape

    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    images = images * std + mean

    # Convert to YCbCr (use luminance channel)
    # Simple RGB to Y conversion: Y = 0.299*R + 0.587*G + 0.114*B
    y_channel = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]

    # Extract 8x8 DCT features from center of image
    center_h, center_w = H // 2, W // 2
    block_size = 8

    features_list = []
    for b in range(B):
        block = y_channel[b, center_h:center_h+block_size, center_w:center_w+block_size]

        # Apply DCT (using FFT as approximation for speed)
        dct = torch.fft.fft2(block).abs()
        features_list.append(dct.flatten())

    features = torch.stack(features_list)
    return features

class FrequencyDetector(nn.Module):
    """DCT-based frequency domain detector"""

    def __init__(self, input_dim=64, hidden_dim=128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 3, 224, 224)

        Returns:
            scores: Tensor of shape (B, 1) with confidence scores
        """
        features = extract_dct_features(x)
        scores = self.mlp(features)
        return scores
