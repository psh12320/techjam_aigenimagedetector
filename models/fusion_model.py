import torch
import torch.nn as nn

class FusionModel(nn.Module):
    """Meta-fusion model that adaptively weights branch predictions"""

    def __init__(self):
        super().__init__()

        # Input: [freq_score, spatial_score, compression, blur, noise, color, resolution]
        # Total: 7 dimensions
        self.fusion_network = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, freq_score, spatial_score, transform_features):
        """
        Args:
            freq_score: Tensor of shape (B, 1)
            spatial_score: Tensor of shape (B, 1)
            transform_features: Tensor of shape (B, 5)

        Returns:
            final_score: Tensor of shape (B, 1)
        """
        # Concatenate all inputs
        combined = torch.cat([freq_score, spatial_score, transform_features], dim=1)

        # Pass through fusion network
        final_score = self.fusion_network(combined)

        return final_score
