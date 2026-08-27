import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.append('.')

from models.frequency_detector import FrequencyDetector
from data.dataset import AIGCDataset
from data.transforms import RobustAugmentation
from training.utils import train_epoch, validate, save_checkpoint

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 20

    print(f"Training on device: {device}")

    # Data - light augmentation for frequency detector
    train_transform = RobustAugmentation(p_augment=0.2, transform_types=['color_jitter', 'crop'])
    val_transform = RobustAugmentation(p_augment=0.0)

    # Replace with actual dataset path
    train_dataset = AIGCDataset('data/processed', split='train', transform=train_transform)
    val_dataset = AIGCDataset('data/processed', split='val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    model = FrequencyDetector().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            save_checkpoint(model, optimizer, epoch, val_loss,
                          'checkpoints/frequency_detector_best.pth', val_metrics)

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()
