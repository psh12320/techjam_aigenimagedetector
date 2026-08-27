import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import sys

from models.frequency_detector import FrequencyDetector
from models.spatial_detector import SpatialDetector
from models.image_analyzer import ImageAnalyzer
from models.fusion_model import FusionModel
from data.transforms import RobustAugmentation
from training.utils import load_checkpoint

class InferenceDataset(Dataset):
    """Dataset for inference on directory of images"""

    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.image_paths = list(self.image_dir.glob('*.[jp][pn]g'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, str(img_path.relative_to(self.image_dir))

def main():
    parser = argparse.ArgumentParser(description='AI-Generated Image Detection Inference')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with images')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--debug', action='store_true', help='Output debug information')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on device: {device}")

    # Load models
    print("Loading models...")
    freq_model = FrequencyDetector().to(device)
    spatial_model = SpatialDetector(pretrained=False).to(device)
    fusion_model = FusionModel().to(device)
    analyzer = ImageAnalyzer()

    load_checkpoint('checkpoints/frequency_detector_best.pth', freq_model)
    load_checkpoint('checkpoints/spatial_detector_best.pth', spatial_model)
    load_checkpoint('checkpoints/fusion_model_best.pth', fusion_model)

    freq_model.eval()
    spatial_model.eval()
    fusion_model.eval()

    # Data
    transform = RobustAugmentation(p_augment=0.0)
    dataset = InferenceDataset(args.input_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Processing {len(dataset)} images...")

    # Inference
    results = []

    with torch.no_grad():
        for images, image_paths in tqdm(dataloader):
            images = images.to(device)

            # Branch predictions
            freq_scores = freq_model(images)
            spatial_scores = spatial_model(images)
            transform_features = analyzer.extract_features(images)

            # Fusion
            final_scores = fusion_model(freq_scores, spatial_scores, transform_features)

            # Collect results
            for i, img_path in enumerate(image_paths):
                result = {
                    'image_path': img_path,
                    'pred': float(final_scores[i].item())
                }

                if args.debug:
                    result['debug'] = {
                        'freq_score': float(freq_scores[i].item()),
                        'spatial_score': float(spatial_scores[i].item()),
                        'transform_features': transform_features[i].cpu().tolist(),
                        'dominant_branch': 'spatial' if spatial_scores[i] > freq_scores[i] else 'frequency'
                    }

                results.append(result)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")
    print(f"Processed {len(results)} images")

    # Summary statistics
    predictions = [r['pred'] for r in results]
    ai_count = sum(1 for p in predictions if p > 0.5)
    real_count = len(predictions) - ai_count
    print(f"Detected: {ai_count} AI-generated, {real_count} real")

if __name__ == '__main__':
    main()
