import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append('.')

from models.frequency_detector import FrequencyDetector
from models.spatial_detector import SpatialDetector
from models.image_analyzer import ImageAnalyzer
from models.fusion_model import FusionModel
from data.dataset import AIGCDataset
from data.transforms import RobustAugmentation
from training.utils import compute_metrics, save_checkpoint, load_checkpoint

def generate_branch_predictions(freq_model, spatial_model, analyzer, dataloader, device):
    """Generate predictions from both branches"""
    freq_model.eval()
    spatial_model.eval()

    all_freq_scores = []
    all_spatial_scores = []
    all_transform_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Generating branch predictions'):
            images = images.to(device)

            freq_scores = freq_model(images)
            spatial_scores = spatial_model(images)
            transform_features = analyzer.extract_features(images)

            all_freq_scores.append(freq_scores.cpu())
            all_spatial_scores.append(spatial_scores.cpu())
            all_transform_features.append(transform_features.cpu())
            all_labels.append(labels)

    return (torch.cat(all_freq_scores),
            torch.cat(all_spatial_scores),
            torch.cat(all_transform_features),
            torch.cat(all_labels))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 15

    print(f"Training fusion model on device: {device}")

    # Load trained branch models
    freq_model = FrequencyDetector().to(device)
    spatial_model = SpatialDetector(pretrained=False).to(device)

    load_checkpoint('checkpoints/frequency_detector_best.pth', freq_model)
    load_checkpoint('checkpoints/spatial_detector_best.pth', spatial_model)

    freq_model.eval()
    spatial_model.eval()

    # Image analyzer
    analyzer = ImageAnalyzer()

    # Data
    val_transform = RobustAugmentation(p_augment=0.0)
    val_dataset = AIGCDataset('data/processed', split='val', transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Generate predictions from branches
    print("Generating branch predictions for validation set...")
    freq_scores, spatial_scores, transform_features, labels = \
        generate_branch_predictions(freq_model, spatial_model, analyzer, val_loader, device)

    # Create fusion training dataset
    fusion_dataset = torch.utils.data.TensorDataset(
        freq_scores, spatial_scores, transform_features, labels.float().unsqueeze(1)
    )

    # Split fusion dataset 80/20 for train/val
    train_size = int(0.8 * len(fusion_dataset))
    val_size = len(fusion_dataset) - train_size
    fusion_train, fusion_val = torch.utils.data.random_split(fusion_dataset, [train_size, val_size])

    train_loader = DataLoader(fusion_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(fusion_val, batch_size=batch_size, shuffle=False)

    # Fusion model
    fusion_model = FusionModel().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training
        fusion_model.train()
        train_loss = 0
        train_predictions = []
        train_labels = []

        for freq, spatial, transform, label in train_loader:
            freq, spatial, transform, label = freq.to(device), spatial.to(device), transform.to(device), label.to(device)

            optimizer.zero_grad()
            output = fusion_model(freq, spatial, transform)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_predictions.append(output.detach().cpu())
            train_labels.append(label.cpu())

        train_loss /= len(train_loader)
        train_predictions = torch.cat(train_predictions)
        train_labels = torch.cat(train_labels)
        train_metrics = compute_metrics(train_predictions, train_labels)

        # Validation
        fusion_model.eval()
        val_loss = 0
        val_predictions = []
        val_labels = []

        with torch.no_grad():
            for freq, spatial, transform, label in val_loader:
                freq, spatial, transform, label = freq.to(device), spatial.to(device), transform.to(device), label.to(device)

                output = fusion_model(freq, spatial, transform)
                loss = criterion(output, label)

                val_loss += loss.item()
                val_predictions.append(output.cpu())
                val_labels.append(label.cpu())

        val_loss /= len(val_loader)
        val_predictions = torch.cat(val_predictions)
        val_labels = torch.cat(val_labels)
        val_metrics = compute_metrics(val_predictions, val_labels)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

        scheduler.step(val_loss)

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            save_checkpoint(fusion_model, optimizer, epoch, val_loss,
                          'checkpoints/fusion_model_best.pth', val_metrics)

    print(f"\nFusion training complete! Best validation accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()
