import torch
from PIL import Image
import numpy as np
import sys
sys.path.append('.')
from data.transforms import get_competition_transforms, RobustAugmentation

def test_jpeg_compression():
    transform = get_competition_transforms('jpeg', quality=50)
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    result = transform(img)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 224, 224)

def test_gaussian_blur():
    transform = get_competition_transforms('blur', sigma=1.0)
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    result = transform(img)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 224, 224)

def test_robust_augmentation():
    aug = RobustAugmentation(p_augment=0.3, transform_types=['jpeg', 'blur'])
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    result = aug(img)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 224, 224)
