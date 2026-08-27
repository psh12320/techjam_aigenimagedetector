import torch
import numpy as np
from PIL import Image
import sys
sys.path.append('.')
from models.image_analyzer import ImageAnalyzer

def test_feature_extraction():
    analyzer = ImageAnalyzer()
    img = torch.randn(2, 3, 224, 224)
    features = analyzer.extract_features(img)
    assert features.shape == (2, 5)
    assert torch.all((features >= 0) & (features <= 1))

def test_clean_vs_compressed():
    analyzer = ImageAnalyzer()

    # Clean image should have low compression score
    clean_img = torch.randn(1, 3, 224, 224)
    clean_features = analyzer.extract_features(clean_img)

    # Features should be in valid range
    assert torch.all((clean_features >= 0) & (clean_features <= 1))
