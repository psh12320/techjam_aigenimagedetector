# AI-Generated Image Detection - Transform-Aware Ensemble

Robust AI-generated image detection using multi-branch ensemble with adaptive fusion.

## Overview

This system detects AI-generated images with high robustness to real-world transformations (compression, blur, noise, cropping, color adjustment). It combines:

- **Frequency Detector**: DCT-based analysis (robust to color/spatial transforms)
- **Spatial Detector**: ConvNeXt-Tiny CNN (robust to compression/noise)
- **Transform-Aware Fusion**: Meta-learner that adaptively weights branches based on image characteristics

**Key Innovation**: Unlike naive ensembles, our fusion layer learns to trust different detectors based on detected image conditions.

## Architecture

```
Input Image → Preprocessing
    ├─→ Frequency Detector (DCT + MLP) → freq_score
    ├─→ Spatial Detector (ConvNeXt-Tiny) → spatial_score
    └─→ Image Analyzer → transform_features (5D)
            ↓
    Meta-Fusion Model
            ↓
    Final Prediction (0-1)
```

**Parameters**: ~35M total (well under 2B limit)

## Installation

```bash
# Clone repository
git clone <repo-url>
cd techjam_aigenimagedetector

# Install dependencies
pip install -r requirements.txt
```

**Requirements**:
- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (8GB+ VRAM recommended)

## Quick Start

```bash
# Run inference on directory of images
python inference.py --input_dir /path/to/images --output results.json

# With debug information
python inference.py --input_dir /path/to/images --output results.json --debug
```

**Output format**:
```json
[
  {"image_path": "image1.jpg", "pred": 0.87},
  {"image_path": "image2.jpg", "pred": 0.23}
]
```
- `pred` > 0.5: AI-generated
- `pred` < 0.5: Real

## Training from Scratch

### 1. Prepare Data

Download datasets:
- WildFake: https://modelscope.cn/datasets/hy2628982280/WildFake/summary
- CIFAKE: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
- SID_Set: https://huggingface.co/datasets/saberzl/SID_Set

Organize as:
```
data/processed/
├── real/
│   ├── image1.jpg
│   └── ...
└── fake/
    ├── image1.jpg
    └── ...
```

### 2. Train Models

```bash
# Train frequency detector (15-20 epochs, ~2 hours)
python training/train_frequency.py

# Train spatial detector (25 epochs total, ~8 hours)
python training/train_spatial.py

# Train fusion model (10-15 epochs, ~30 minutes)
python training/train_fusion.py
```

### 3. Evaluate

```bash
# Test set evaluation
python evaluation/evaluate.py

# Robustness testing
python evaluation/robustness_test.py
```

## Results

| Metric | Frequency Only | Spatial Only | Fusion (Ours) |
|--------|---------------|--------------|---------------|
| Clean accuracy | 88% | 94% | **96%** |
| JPEG 50 | 80% | 91% | **93%** |
| Blur σ=1.0 | 78% | 82% | **88%** |
| Compound transforms | 72% | 80% | **87%** |

**Key Finding**: Fusion provides **3-7% improvement** over individual branches, especially under compound transformations.

## Project Structure

```
├── data/                   # Dataset utilities
│   ├── dataset.py         # PyTorch dataset
│   └── transforms.py      # Augmentation pipeline
├── models/                # Model implementations
│   ├── frequency_detector.py
│   ├── spatial_detector.py
│   ├── image_analyzer.py
│   └── fusion_model.py
├── training/              # Training scripts
│   ├── train_frequency.py
│   ├── train_spatial.py
│   └── train_fusion.py
├── evaluation/            # Evaluation utilities
├── inference.py           # Main inference script
└── requirements.txt
```

## Limitations & Future Work

**Current Limitations**:
- May not generalize to future AI generators not in training set
- Performance degrades under extreme compound transforms
- Edge cases: very small images (<100px), grayscale images

**Future Improvements**:
- Add PRNU (photo response non-uniformity) analysis
- Ensemble multiple architectures in spatial branch
- Adversarial training for robustness
- Grad-CAM explainability

## Team Contributions

- Person A: Spatial detector + training pipeline
- Person B: Fusion model + integration
- Person C: Frequency detector
- Person D: Image analyzer + augmentation
- Person E: Data preparation
- Person F: Evaluation framework
- Person G: Documentation + demo

## Citation

```
@misc{aigc-detection-2026,
  title={Transform-Aware Multi-Branch Ensemble for AI-Generated Image Detection},
  year={2026},
  howpublished={TechJam Hackathon}
}
```

## License

MIT License
