import torch
import torch.nn as nn
import sys
sys.path.append('.')
from training.utils import compute_metrics, save_checkpoint, load_checkpoint
import os

def test_compute_metrics():
    predictions = torch.tensor([0.9, 0.8, 0.3, 0.2])
    labels = torch.tensor([1, 1, 0, 0])

    metrics = compute_metrics(predictions, labels, threshold=0.5)

    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert metrics['accuracy'] == 1.0

def test_checkpoint_save_load():
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters())

    checkpoint_path = 'test_checkpoint.pth'
    save_checkpoint(model, optimizer, epoch=5, loss=0.5, path=checkpoint_path)

    assert os.path.exists(checkpoint_path)

    loaded_data = load_checkpoint(checkpoint_path, model, optimizer)
    assert loaded_data['epoch'] == 5
    assert loaded_data['loss'] == 0.5

    os.remove(checkpoint_path)
