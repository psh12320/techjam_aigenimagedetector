import torch
import sys
sys.path.append('.')
from models.fusion_model import FusionModel

def test_fusion_model():
    model = FusionModel()

    freq_scores = torch.rand(2, 1)
    spatial_scores = torch.rand(2, 1)
    transform_features = torch.rand(2, 5)

    output = model(freq_scores, spatial_scores, transform_features)
    assert output.shape == (2, 1)
    assert torch.all((output >= 0) & (output <= 1))

def test_parameter_count():
    model = FusionModel()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 10_000_000  # Should be around 5M
