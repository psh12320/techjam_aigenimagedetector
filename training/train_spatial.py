import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.append('.')

from models.spatial_detector import SpatialDetector
from data.dataset import AIGCDataset
from data.transforms import RobustAugmentation
from training.utils import train_epoch, validate, save_checkpoint

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_epochs_phase1 = 5
    num_epochs_phase2 = 20

    print(f"Training on device: {device}")

    # Data - heavy augmentation for spatial detector
    train_transform = RobustAugmentation(p_augment=0.3)
    val_transform = RobustAugmentation(p_augment=0.0)

    train_dataset = AIGCDataset('data/processed', split='train', transform=train_transform)
    val_dataset = AIGCDataset('data/processed', split='val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    model = SpatialDetector(pretrained=True).to(device)
    criterion = nn.BCELoss()

    # Phase 1: Train head only
    print("\n=== Phase 1: Training classification head (backbone frozen) ===")
    model.freeze_backbone()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

    best_val_acc = 0.0

    for epoch in range(num_epochs_phase1):
        print(f"\nEpoch {epoch+1}/{num_epochs_phase1}")

        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']

    # Phase 2: Fine-tune entire model
    print("\n=== Phase 2: Fine-tuning entire model ===")
    model.unfreeze_backbone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_phase2)

    for epoch in range(num_epochs_phase2):
        print(f"\nEpoch {epoch+1}/{num_epochs_phase2}")

        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

        scheduler.step()

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            save_checkpoint(model, optimizer, epoch, val_loss,
                          'checkpoints/spatial_detector_best.pth', val_metrics)

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()
