import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('.')

from models.frequency_detector import FrequencyDetector
from models.spatial_detector import SpatialDetector
from models.image_analyzer import ImageAnalyzer
from models.fusion_model import FusionModel
from data.dataset import AIGCDataset
from data.transforms import RobustAugmentation
from training.utils import load_checkpoint, compute_metrics

def evaluate_model(model, dataloader, device, model_name="Model"):
    """Evaluate a single model"""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)

            all_predictions.append(outputs.cpu())
            all_labels.append(labels)

    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)

    metrics = compute_metrics(predictions, labels)

    print(f"\n{model_name} Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc']:.4f}")

    return metrics

def evaluate_ensemble(freq_model, spatial_model, fusion_model, analyzer, dataloader, device):
    """Evaluate full ensemble"""
    freq_model.eval()
    spatial_model.eval()
    fusion_model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            freq_scores = freq_model(images)
            spatial_scores = spatial_model(images)
            transform_features = analyzer.extract_features(images)
            final_scores = fusion_model(freq_scores, spatial_scores, transform_features)

            all_predictions.append(final_scores.cpu())
            all_labels.append(labels)

    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)

    metrics = compute_metrics(predictions, labels)

    print(f"\nEnsemble (Fusion) Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc']:.4f}")

    return metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32

    # Load models
    freq_model = FrequencyDetector().to(device)
    spatial_model = SpatialDetector(pretrained=False).to(device)
    fusion_model = FusionModel().to(device)
    analyzer = ImageAnalyzer()

    load_checkpoint('checkpoints/frequency_detector_best.pth', freq_model)
    load_checkpoint('checkpoints/spatial_detector_best.pth', spatial_model)
    load_checkpoint('checkpoints/fusion_model_best.pth', fusion_model)

    # Test data
    test_transform = RobustAugmentation(p_augment=0.0)
    test_dataset = AIGCDataset('data/processed', split='test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Evaluating on {len(test_dataset)} test images")

    # Evaluate individual branches
    freq_metrics = evaluate_model(freq_model, test_loader, device, "Frequency Detector")
    spatial_metrics = evaluate_model(spatial_model, test_loader, device, "Spatial Detector")

    # Evaluate ensemble
    ensemble_metrics = evaluate_ensemble(freq_model, spatial_model, fusion_model,
                                        analyzer, test_loader, device)

    # Compare
    print("\n=== Comparison ===")
    print(f"Frequency:  {freq_metrics['accuracy']:.4f}")
    print(f"Spatial:    {spatial_metrics['accuracy']:.4f}")
    print(f"Fusion:     {ensemble_metrics['accuracy']:.4f}")

    improvement = ensemble_metrics['accuracy'] - max(freq_metrics['accuracy'], spatial_metrics['accuracy'])
    print(f"Fusion improvement: +{improvement:.4f} ({improvement*100:.1f}%)")

if __name__ == '__main__':
    main()
