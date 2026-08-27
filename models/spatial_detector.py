import torch
import torch.nn as nn
import timm

class SpatialDetector(nn.Module):
    """ConvNeXt-Tiny based spatial detector"""

    def __init__(self, pretrained=True):
        super().__init__()

        # Load ConvNeXt-Tiny with ImageNet pretrained weights
        self.backbone = timm.create_model('convnext_tiny', pretrained=pretrained, num_classes=0)

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]

        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 3, 224, 224)

        Returns:
            scores: Tensor of shape (B, 1) with confidence scores
        """
        features = self.backbone(x)
        scores = self.classifier(features)
        return scores

    def freeze_backbone(self):
        """Freeze backbone for phase 1 training"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for phase 2 training"""
        for param in self.backbone.parameters():
            param.requires_grad = True
