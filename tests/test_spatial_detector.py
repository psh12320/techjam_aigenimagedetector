import torch
import sys
sys.path.append('.')
from models.spatial_detector import SpatialDetector

def test_spatial_detector():
    model = SpatialDetector(pretrained=False)
    img = torch.randn(2, 3, 224, 224)
    output = model(img)
    assert output.shape == (2, 1)
    assert torch.all((output >= 0) & (output <= 1))

def test_parameter_count():
    model = SpatialDetector(pretrained=False)
    n_params = sum(p.numel() for p in model.parameters())
    assert 25_000_000 < n_params < 35_000_000  # ConvNeXt-Tiny is ~28M

def test_pretrained_loading():
    model = SpatialDetector(pretrained=True)
    assert model is not None
