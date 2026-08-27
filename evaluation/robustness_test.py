import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('.')

from models.frequency_detector import FrequencyDetector
from models.spatial_detector import SpatialDetector
from models.image_analyzer import ImageAnalyzer
from models.fusion_model import FusionModel
from data.dataset import AIGCDataset
from data.transforms import get_competition_transforms
from training.utils import load_checkpoint, compute_metrics

def test_transform_robustness(freq_model, spatial_model, fusion_model, analyzer,
                              dataset, device, transform_type, **kwargs):
    """Test robustness to specific transform"""
    transform = get_competition_transforms(transform_type, **kwargs)
    test_dataset = AIGCDataset(dataset.root, split='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Evaluate ensemble
    freq_model.eval()
    spatial_model.eval()
    fusion_model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            freq_scores = freq_model(images)
            spatial_scores = spatial_model(images)
            transform_features = analyzer.extract_features(images)
            final_scores = fusion_model(freq_scores, spatial_scores, transform_features)

            all_predictions.append(final_scores.cpu())
            all_labels.append(labels)

    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)

    return compute_metrics(predictions, labels)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    freq_model = FrequencyDetector().to(device)
    spatial_model = SpatialDetector(pretrained=False).to(device)
    fusion_model = FusionModel().to(device)
    analyzer = ImageAnalyzer()

    load_checkpoint('checkpoints/frequency_detector_best.pth', freq_model)
    load_checkpoint('checkpoints/spatial_detector_best.pth', spatial_model)
    load_checkpoint('checkpoints/fusion_model_best.pth', fusion_model)

    # Test dataset
    from data.transforms import RobustAugmentation
    clean_transform = RobustAugmentation(p_augment=0.0)
    test_dataset = AIGCDataset('data/processed', split='test', transform=clean_transform)

    # Baseline (clean)
    print("Testing robustness to transformations...\n")

    transforms_to_test = [
        ('Clean', {}),
        ('JPEG 90', {'transform_type': 'jpeg', 'quality': 90}),
        ('JPEG 70', {'transform_type': 'jpeg', 'quality': 70}),
        ('JPEG 50', {'transform_type': 'jpeg', 'quality': 50}),
        ('JPEG 30', {'transform_type': 'jpeg', 'quality': 30}),
        ('Blur 0.5', {'transform_type': 'blur', 'sigma': 0.5}),
        ('Blur 1.0', {'transform_type': 'blur', 'sigma': 1.0}),
        ('Blur 2.0', {'transform_type': 'blur', 'sigma': 2.0}),
        ('Noise 0.02', {'transform_type': 'noise', 'sigma': 0.02}),
        ('Noise 0.05', {'transform_type': 'noise', 'sigma': 0.05}),
        ('Color Jitter', {'transform_type': 'color_jitter'}),
        ('Crop 80%', {'transform_type': 'crop'}),
    ]

    results = []

    for name, params in transforms_to_test:
        if name == 'Clean':
            metrics = test_transform_robustness(freq_model, spatial_model, fusion_model,
                                               analyzer, test_dataset, device, 'clean')
        else:
            metrics = test_transform_robustness(freq_model, spatial_model, fusion_model,
                                               analyzer, test_dataset, device, **params)

        results.append((name, metrics['accuracy']))
        print(f"{name:15s}: Accuracy = {metrics['accuracy']:.4f}")

    # Summary
    print("\n=== Robustness Summary ===")
    clean_acc = results[0][1]
    print(f"Clean accuracy: {clean_acc:.4f}")

    avg_degradation = sum(clean_acc - acc for _, acc in results[1:]) / (len(results) - 1)
    print(f"Average degradation: {avg_degradation:.4f} ({avg_degradation*100:.1f}%)")

if __name__ == '__main__':
    main()
