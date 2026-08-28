import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import os

def compute_metrics(predictions, labels, threshold=0.5):
    """Compute classification metrics

    Args:
        predictions: Tensor of shape (N,) with probabilities
        labels: Tensor of shape (N,) with binary labels
        threshold: Classification threshold

    Returns:
        metrics: Dictionary with accuracy, precision, recall, f1, auc
    """
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    pred_binary = (predictions >= threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(labels, pred_binary),
        'precision': precision_score(labels, pred_binary, zero_division=0),
        'recall': recall_score(labels, pred_binary, zero_division=0),
        'f1': f1_score(labels, pred_binary, zero_division=0),
        'auc': roc_auc_score(labels, predictions) if len(np.unique(labels)) > 1 else 0.0
    }

    return metrics

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch

    Returns:
        avg_loss: Average loss for the epoch
        metrics: Training metrics
    """
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc='Training'):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_predictions.append(outputs.detach())
        all_labels.append(labels.detach())

    avg_loss = total_loss / len(dataloader)

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_predictions, all_labels)

    return avg_loss, metrics

def validate(model, dataloader, criterion, device):
    """Validate model

    Returns:
        avg_loss: Average validation loss
        metrics: Validation metrics
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_predictions.append(outputs)
            all_labels.append(labels)

    avg_loss = total_loss / len(dataloader)

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_predictions, all_labels)

    return avg_loss, metrics

def save_checkpoint(model, optimizer, epoch, loss, path, metrics=None):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(path, model, optimizer=None):
    """Load training checkpoint

    Returns:
        Dictionary with epoch, loss, metrics
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss'],
        'metrics': checkpoint.get('metrics', {})
    }
