import torch
import sys
sys.path.append('.')
from models.frequency_detector import FrequencyDetector, extract_dct_features

def test_dct_extraction():
    img = torch.randn(2, 3, 224, 224)
    features = extract_dct_features(img)
    assert features.shape[0] == 2
    assert features.shape[1] == 64

def test_frequency_detector():
    model = FrequencyDetector()
    img = torch.randn(2, 3, 224, 224)
    output = model(img)
    assert output.shape == (2, 1)
    assert torch.all((output >= 0) & (output <= 1))

def test_parameter_count():
    model = FrequencyDetector()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 3_000_000  # Should be around 2M
