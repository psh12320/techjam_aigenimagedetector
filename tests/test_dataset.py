import torch
from PIL import Image
import numpy as np
import os
import sys
sys.path.append('.')
from data.dataset import AIGCDataset

def setup_dummy_data():
    """Create dummy dataset for testing"""
    os.makedirs('data/test_images/real', exist_ok=True)
    os.makedirs('data/test_images/fake', exist_ok=True)

    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(f'data/test_images/real/img_{i}.jpg')
        img.save(f'data/test_images/fake/img_{i}.jpg')

def test_dataset_loading():
    setup_dummy_data()
    dataset = AIGCDataset('data/test_images', split='train')
    assert len(dataset) > 0

    img, label = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert label in [0, 1]

def test_dataset_labels():
    setup_dummy_data()
    dataset = AIGCDataset('data/test_images', split='train')

    # Check we have both classes
    labels = [dataset[i][1] for i in range(len(dataset))]
    assert 0 in labels
    assert 1 in labels
