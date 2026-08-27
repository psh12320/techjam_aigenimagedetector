# AI-Generated Image Detection System - Design Document

**Date:** 2026-08-27  
**Competition:** TechJam Hackathon - AI-Generated Image Detection Challenge  
**Deadline:** September 1, 2026  
**Team:** 5 members (2 model specialists, 2 CV experts, 1 support)

## Executive Summary

We propose a **Transform-Aware Multi-Branch Ensemble** for robust AI-generated image detection. The system combines frequency domain analysis, deep spatial feature learning, and intelligent fusion that adapts predictions based on detected image transformations. This approach directly addresses the competition's core challenge: maintaining detection accuracy under real-world image transformations (compression, blur, noise, cropping, color adjustment).

**Key Innovation:** Unlike naive ensembles, our meta-fusion layer learns to weight detection branches based on image characteristics—trusting frequency analysis more when blur is detected, and spatial features more when compression is present.

## Problem Context

### Competition Requirements
- **Goal:** Detect AI-generated images with robustness to real-world transformations
- **Constraints:** 
  - Model <2B parameters
  - Transforms: JPEG compression, Gaussian blur, resize, noise, color jitter, center crop
  - 3-day implementation timeline
- **Deliverables:** Inference script (image dir → JSON predictions), code repository, demo video
- **Judging:** Technical execution (35%), Innovation (20%), Impact (20%), Feasibility (15%), Presentation (10%)

### Core Insight
Different detection methods have different robustness profiles:
- **Frequency analysis:** Robust to color/spatial transforms, struggles with heavy blur/compression
- **Deep spatial features:** Robust to compression/noise, struggles with blur
- **Opportunity:** Adaptive fusion that selects the best detector for each image's condition

## System Architecture

### High-Level Flow
```
Input Image
    ↓
Preprocessing (resize, normalize)
    ↓
    ├─→ Branch 1: Frequency Detector → freq_score (0-1)
    ├─→ Branch 2: Spatial Detector → spatial_score (0-1)
    ├─→ Image Analyzer → transform_features (5D vector)
    ↓
Meta-Fusion Model
    ↓
Final Prediction (0-1)
```

### Component Overview

| Component | Purpose | Parameters | Key Strength |
|-----------|---------|------------|--------------|
| Frequency Detector | DCT-based artifact detection | ~2M (MLP) | Robust to color/crop |
| Spatial Detector | Deep CNN features | 28M (ConvNeXt-Tiny) | Robust to compression |
| Image Analyzer | Transform detection | 0 (feature extraction) | Informs fusion |
| Meta-Fusion | Adaptive combination | ~5M (small MLP) | Smart branch weighting |
| **Total** | **Complete system** | **~35M** | **Well under 2B limit** |

## Component Specifications

### Branch 1: Frequency Detector

**Approach:** Discrete Cosine Transform (DCT) analysis of frequency domain artifacts

**Implementation:**
1. Convert image to YCbCr color space (luminance + chrominance)
2. Extract DCT coefficients from Y channel (8×8 blocks)
3. Analyze high-frequency components (where AI generators leave artifacts)
4. Feed flattened frequency features to 3-layer MLP
5. Output: confidence score 0-1

**Architecture:**
```python
DCT Feature Extractor (parameter-free)
    ↓
Flatten → [batch, 64] frequency features
    ↓
MLP: 64 → 128 → 64 → 1
    ↓
Sigmoid → confidence score
```

**Training:**
- Dataset: WildFake + CIFAKE + SID_Set
- Augmentation: Light (color jitter, crop) - preserves frequency structure
- Loss: Binary cross-entropy
- Epochs: 15-20
- Learning rate: 1e-3

**Rationale:**
- JPEG compression operates in DCT space → natural robustness
- AI generators produce subtle frequency anomalies
- Fast inference (~10ms per image)
- Explainable (can visualize triggering frequencies)

**Expected Performance:**
- Clean images: 85-90% accuracy
- Color jitter / crop: 80-85% accuracy (minimal degradation)
- Heavy blur: 65-70% accuracy (frequency smoothing)

### Branch 2: Deep Spatial Detector

**Approach:** Fine-tuned ConvNeXt-Tiny CNN for spatial artifact detection

**Model Choice: ConvNeXt-Tiny**
- Modern CNN architecture (Facebook AI, 2022)
- 28M parameters (well under 2B limit)
- ImageNet pretrained weights
- Superior to ViTs for local artifact detection

**Implementation:**
1. Replace classification head (1000 classes → binary)
2. Input: 224×224 RGB images
3. Backbone extracts hierarchical spatial features
4. Classification head outputs confidence 0-1

**Training Strategy:**
```
Phase 1 (5 epochs):
- Freeze backbone
- Train new classification head only
- Learning rate: 1e-3
- Warm up new parameters

Phase 2 (15-20 epochs):
- Unfreeze entire model
- Fine-tune end-to-end
- Learning rate: 1e-4 with cosine decay
- Heavy augmentation (see Data Pipeline)

Optimizer: AdamW (weight decay 1e-4)
Loss: Binary cross-entropy
```

**What It Learns:**
- Spatial artifacts (unnatural edges, texture inconsistencies)
- Semantic implausibilities (weird object relationships)
- Generator-specific fingerprints (GAN artifacts, diffusion patterns)

**Expected Performance:**
- Clean images: 92-95% accuracy
- JPEG compression: 90-93% accuracy (minor degradation)
- Heavy blur: 75-80% accuracy (loses fine details)

### Image Analyzer: Transform Detection

**Purpose:** Extract image characteristics to inform fusion layer

**Features Extracted:**

1. **Compression Artifacts Score (0-1)**
   - Detect DCT quantization patterns
   - Check for 8×8 block boundaries
   - Measure compression noise variance

2. **Blur Estimation (0-1)**
   - Laplacian variance method
   - High variance = sharp, low = blurry
   - Normalize to 0-1 scale

3. **Noise Level (0-1)**
   - Estimate Gaussian noise standard deviation
   - Use median absolute deviation in homogeneous regions

4. **Color Distribution Stats (0-1)**
   - Saturation variance
   - Detect aggressive color jitter
   - HSV histogram analysis

5. **Resolution Consistency (0-1)**
   - Check for upscaling artifacts
   - Frequency domain analysis for interpolation patterns

**Output:** 5-dimensional feature vector

**Implementation:** Simple feature extraction functions (OpenCV + NumPy)
- No trainable parameters
- Fast execution (<5ms per image)
- Robust and deterministic

**Example Values:**
- Clean image: [0.1, 0.9, 0.05, 0.5, 0.95]
- Compressed: [0.8, 0.85, 0.1, 0.5, 0.9]
- Blurred: [0.2, 0.3, 0.05, 0.5, 0.95]

### Meta-Fusion Model: Transform-Aware Combination

**Architecture:**

```python
Input Layer: [freq_score, spatial_score, compression, blur, noise, color, resolution]
    ↓ (7-dimensional input)
Hidden Layer 1: 64 neurons + ReLU + Dropout(0.3)
    ↓
Hidden Layer 2: 32 neurons + ReLU + Dropout(0.2)
    ↓
Hidden Layer 3: 16 neurons + ReLU
    ↓
Output Layer: 1 neuron + Sigmoid
    ↓
Final Confidence Score (0-1)
```

**Parameters:** ~5M (small, fast to train)

**Training Strategy:**
1. **Prerequisite:** Branch 1 and Branch 2 must be trained first
2. **Freeze both branches** (inference mode only)
3. Generate predictions from both branches on validation set
4. Train fusion model to predict ground truth from:
   - Branch outputs (2 values)
   - Transform features (5 values)
5. **Loss:** Binary cross-entropy
6. **Epochs:** 10-15 (fast convergence)
7. **Learning rate:** 1e-3 with ReduceLROnPlateau

**What It Learns:**

The fusion model learns decision rules like:
- `if compression_score > 0.7: weight_spatial = 0.7, weight_freq = 0.3`
- `if blur_score > 0.6: weight_freq = 0.6, weight_spatial = 0.4`
- `if clean_image: balanced_mix or learned_optimal`

But crucially, it learns **non-linear combinations** beyond simple weighted averages.

**Example Learned Behavior:**
```python
# Pseudocode representation
if blur > 0.6 and noise < 0.2:
    trust frequency_detector more
elif compression > 0.7:
    trust spatial_detector more
elif both_agree (|freq_score - spatial_score| < 0.2):
    high_confidence prediction
elif both_disagree (|freq_score - spatial_score| > 0.7):
    conservative prediction (closer to 0.5)
```

**Explainability:**
- Can compute branch contribution: which branch dominated the decision
- Useful for error analysis and presentation

**Innovation Claim:** This adaptive fusion is our key differentiator—not a naive ensemble, but intelligent collaboration based on image conditions.

## Data Pipeline

### Datasets

**Primary:**
- **WildFake** ([ModelScope](https://modelscope.cn/datasets/hy2628982280/WildFake/summary))
  - Real-world diverse generators
  - Multiple AI generation methods
  - Best for generalization

**Secondary:**
- **CIFAKE** ([Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images))
  - CIFAR-style paired data
  - Clean real vs AI labels
  
- **SID_Set** ([HuggingFace](https://huggingface.co/datasets/saberzl/SID_Set))
  - Additional diversity

**Validation:**
- **COCO val2017:** 4,998 non-AIGC images
- **DALL·E Advanced:** 8,843 AIGC images
- Note: This is for tracking progress only, not final scoring

### Data Split Strategy

```
Training Set: 80% of combined dataset
    ↓
Used for Branch 1 and Branch 2 training

Validation Set: 10% of combined dataset + provided COCO/DALL·E
    ↓
Used for hyperparameter tuning and fusion model training

Test Set: 10% of combined dataset (held out)
    ↓
Final evaluation before submission
```

### Augmentation Pipeline

**Critical for Robustness:** Train with the exact transforms the test set will use.

**Transform Specification (matches competition requirements):**
```python
1. JPEG Compression: quality ∈ {90, 70, 50, 30}
2. Gaussian Blur: σ ∈ {0.5, 1.0, 2.0}
3. Resize: scale to 0.5× or 0.25×, then upscale to original
4. Gaussian Noise: σ ∈ {0.02, 0.05, 0.10}
5. Color Jitter: brightness/contrast/saturation ± 20%
6. Center Crop: crop to 80% then resize back
```

**Augmentation Strategy:**

**For Branch 2 (Spatial Detector):**
- 70% of batches: clean images (prevent overfitting to augmented data)
- 30% of batches: apply 1-3 random transforms from above
- Always apply basic augmentation: horizontal flip, small rotation (±10°)

**For Branch 1 (Frequency Detector):**
- 80% clean (frequency analysis is inherently more sensitive)
- 20% augmented with color/crop only (preserve frequency structure)

**Rationale:**
- Training heavily on augmented data causes models to "expect" degradation
- Balancing clean and transformed data maintains both accuracy and robustness

### Data Loading

**Framework:** PyTorch DataLoader
- Batch size: 32 (adjust based on GPU memory)
- Num workers: 4 (parallel data loading)
- Pin memory: True (faster GPU transfer)
- Prefetch factor: 2

**Preprocessing:**
1. Resize to 224×224 (ConvNeXt-Tiny input size)
2. Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] (ImageNet stats)
3. Convert to torch tensors

## Implementation Plan Overview

### Project Structure
```
techjam_aigenimagedetector/
├── data/
│   ├── raw/                    # Downloaded datasets
│   ├── processed/              # Preprocessed + split
│   └── validation/             # COCO + DALL·E validation set
├── models/
│   ├── frequency_detector.py  # Branch 1 implementation
│   ├── spatial_detector.py    # Branch 2 implementation
│   ├── fusion_model.py        # Meta-fusion implementation
│   ├── image_analyzer.py      # Transform detection
│   └── ensemble.py            # Full system integration
├── training/
│   ├── train_frequency.py     # Train Branch 1
│   ├── train_spatial.py       # Train Branch 2
│   ├── train_fusion.py        # Train fusion model
│   └── utils.py               # Training utilities
├── evaluation/
│   ├── evaluate.py            # Validation metrics
│   ├── robustness_test.py     # Test on transforms
│   └── error_analysis.py      # Analyze failures
├── inference.py               # Main submission script
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
└── docs/
    └── presentation/          # Demo video materials
```

### Timeline (3 Days)

**Day 1: Data + Individual Branches**
- Morning: Download datasets, setup environment
- Afternoon: Implement and train Branch 1 (Frequency)
- Evening: Implement and train Branch 2 (Spatial)
- Goal: Both branches working with baseline metrics

**Day 2: Fusion + Robustness Testing**
- Morning: Implement Image Analyzer + Meta-Fusion
- Afternoon: Train fusion model
- Evening: Robustness evaluation on all transforms
- Goal: Full system working, identify weaknesses

**Day 3: Refinement + Presentation**
- Morning: Error analysis, final training tweaks
- Afternoon: Prepare demo video, README, code cleanup
- Evening: Submit to Devpost
- Goal: Polished submission ready

### Division of Labor

**Model Team (2 people):**
- Person A: Branch 2 (Spatial Detector) + training pipeline
- Person B: Meta-Fusion model + integration

**CV Team (2 people):**
- Person C: Branch 1 (Frequency Detector)
- Person D: Image Analyzer + augmentation pipeline

**Support Team (3 people):**
- Person E: Data preparation and management
- Person F: Evaluation framework + error analysis
- Person G: Demo video + presentation materials

**Daily Standups:** Quick sync on progress, blockers, and next steps

## Inference Pipeline

### Main Script: `inference.py`

**Command Line Interface:**
```bash
python inference.py --input_dir /path/to/images --output results.json [--batch_size 32] [--device cuda]
```

**Processing Flow:**
```python
1. Load pretrained models:
   - frequency_detector.pth
   - spatial_detector.pth
   - fusion_model.pth

2. Scan input directory for images (jpg, jpeg, png)

3. For each batch of images:
   a. Preprocess (resize, normalize)
   b. Branch 1: freq_score = frequency_detector(image)
   c. Branch 2: spatial_score = spatial_detector(image)
   d. Analyzer: transform_features = image_analyzer(image)
   e. Fusion: final_pred = fusion_model([freq_score, spatial_score, transform_features])

4. Collect results:
   results = [
       {"image_path": "path/to/img1.jpg", "pred": 0.87},
       {"image_path": "path/to/img2.jpg", "pred": 0.23},
       ...
   ]

5. Write JSON output
```

**Output Format (Competition Requirement):**
```json
[
  {
    "image_path": "relative/path/to/image1.jpg",
    "pred": 0.8734
  },
  {
    "image_path": "relative/path/to/image2.jpg",
    "pred": 0.1256
  }
]
```

**Performance Targets:**
- **Speed:** <100ms per image on GPU (batch processing)
- **Memory:** <4GB GPU RAM
- **Scalability:** Can process 10,000 images without issues

**Optional Debug Mode:**
```bash
python inference.py --input_dir images/ --output results.json --debug
```
Outputs additional information:
```json
{
  "image_path": "image1.jpg",
  "pred": 0.87,
  "debug": {
    "freq_score": 0.82,
    "spatial_score": 0.91,
    "transform_features": [0.6, 0.3, 0.1, 0.5, 0.8],
    "dominant_branch": "spatial"
  }
}
```

## Evaluation Strategy

### Metrics

**Primary Metrics:**
1. **Accuracy:** (TP + TN) / Total
2. **Precision:** TP / (TP + FP) — low false positives important
3. **Recall:** TP / (TP + FN) — catch real AI images
4. **F1 Score:** Harmonic mean of precision and recall
5. **AUC-ROC:** Area under receiver operating characteristic curve

**Threshold:** 0.5 for binary classification (tune if needed)

### Evaluation Protocol

**1. Baseline Performance (Clean Validation Set)**
```python
Test on COCO val2017 + DALL·E Advanced (no transforms)
Expected: >95% accuracy
```

**2. Robustness Testing (Individual Transforms)**
Apply each transform type separately and measure performance:
```python
for transform in [JPEG90, JPEG70, JPEG50, JPEG30, Blur0.5, Blur1.0, ...]:
    transformed_images = apply_transform(validation_set, transform)
    metrics = evaluate(model, transformed_images)
    degradation = baseline_accuracy - metrics.accuracy
```

**Target:** <10% accuracy drop for single transforms

**3. Compound Transforms Testing**
Apply multiple transforms (real-world scenario):
```python
# Example: Compressed + blurred + color jittered
combined_transforms = [
    [JPEG50, Blur1.0],
    [JPEG30, Noise0.05, ColorJitter],
    [Resize0.5, Crop80, JPEG70]
]
```

**Target:** >85% accuracy on 2-3 combined transforms

**4. Per-Branch Analysis**
```python
Compare:
- Branch 1 only (frequency)
- Branch 2 only (spatial)
- Fusion (full system)

Validate fusion provides ≥3% improvement over best individual branch
```

**5. Error Analysis**

**False Positives (Real images classified as AI):**
- What characteristics do they share?
- Over-processed photos? Heavy filters?

**False Negatives (AI images classified as Real):**
- Which generators are hardest to detect?
- What transforms break detection most?

**Confusion Matrix Visualization:**
```
                Predicted
              Real    AI
Actual  Real  [TN]   [FP]
        AI    [FN]   [TP]
```

### Success Criteria

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| Clean validation accuracy | >95% | >97% |
| Single transform accuracy | >90% | >92% |
| Compound transform accuracy | >85% | >88% |
| Fusion improvement | >3% | >5% |
| Inference speed | <100ms/img | <50ms/img |

### Validation Set Usage

**Important:** The provided validation set (COCO + DALL·E) is for demonstration only and does not contribute to final score.

**Our Strategy:**
- Use it for tracking progress and debugging
- Create our own held-out test set from training data
- Report both metrics in presentation

## Robustness Analysis

### Expected Performance by Transform Type

| Transform Type | Branch 1 (Freq) | Branch 2 (Spatial) | Fusion (Expected) |
|----------------|-----------------|---------------------|-------------------|
| Clean | 88% | 94% | 96% |
| JPEG 90 | 86% | 93% | 95% |
| JPEG 50 | 80% | 91% | 93% |
| JPEG 30 | 72% | 89% | 90% |
| Blur σ=0.5 | 85% | 90% | 92% |
| Blur σ=1.0 | 78% | 82% | 88% |
| Blur σ=2.0 | 65% | 70% | 82% |
| Noise σ=0.05 | 84% | 92% | 94% |
| Color Jitter | 87% | 91% | 94% |
| Crop 80% | 86% | 90% | 93% |

**Key Observations:**
- Heavy compression hurts frequency more than spatial
- Heavy blur hurts both, but fusion compensates
- Fusion consistently outperforms individual branches

### Failure Modes & Mitigations

**1. Novel AI Generators (not in training set)**
- **Risk:** Models overfit to known generators
- **Mitigation:** 
  - Train on diverse dataset (WildFake has multiple generators)
  - Frequency analysis generalizes better (universal artifacts)
  - Test on held-out generators if available

**2. Extreme Compound Transforms**
- **Risk:** Heavy blur + compression + noise destroys all signals
- **Mitigation:** 
  - Graceful degradation acceptable
  - Document limitations clearly
  - Competition likely doesn't test extreme cases

**3. Edge Cases**
- **Very small images** (<100px): Resize may lose information
- **Black & white images:** Color jitter features break
- **Extreme crops:** May miss generator artifacts
- **Mitigation:** Add input validation, handle gracefully, document

**4. Adversarial Examples**
- **Risk:** Someone intentionally tries to fool the detector
- **Mitigation:** Out of scope for this competition, but mention as future work

## Technical Stack

### Dependencies

```
Core:
- Python 3.10+
- PyTorch 2.0+
- torchvision 0.15+
- timm 0.9+ (PyTorch Image Models - for ConvNeXt)

Computer Vision:
- opencv-python 4.8+
- Pillow 10.0+
- scikit-image 0.21+

Data & ML:
- numpy 1.24+
- pandas 2.0+
- scikit-learn 1.3+

Utilities:
- tqdm (progress bars)
- matplotlib (visualization)
- seaborn (plotting)

Optional:
- wandb (experiment tracking)
- albumentations (augmentation library)
```

### Hardware Requirements

**Minimum:**
- GPU: 8GB VRAM (e.g., RTX 3070, T4)
- RAM: 16GB
- Storage: 50GB for datasets + models

**Recommended:**
- GPU: 16GB VRAM (e.g., RTX 4090, A100)
- RAM: 32GB
- Storage: 100GB

**Cloud Alternatives:**
- Google Colab Pro (A100 GPU)
- Kaggle Notebooks (30h GPU/week)
- AWS SageMaker (if budget available)

## Presentation & Demo Strategy

### Demo Video Structure (3-5 minutes)

**1. Hook (15 seconds)**
- Show two images side by side: "Which one is AI-generated?"
- Reveal answer: "AI images are everywhere, but detection breaks after compression or editing"

**2. Problem Deep Dive (20 seconds)**
- Show same AI image: clean → compressed → blurred
- Simple detector: 95% → 78% → 65% accuracy
- "We need robustness, not just accuracy"

**3. Our Insight (20 seconds)**
- "We discovered different detectors fail differently"
- Frequency analysis: robust to color, breaks with blur
- Spatial analysis: robust to compression, breaks with blur
- "So we made them collaborate"

**4. Architecture Walkthrough (60 seconds)**
- Diagram of 3-branch system
- Branch 1: Frequency detector (show DCT visualization)
- Branch 2: Spatial detector (show CNN heatmap)
- Fusion: "Smart combination based on image condition"
- Animation: compressed image → fusion trusts spatial more

**5. Live Demo (90 seconds)**
```bash
# Terminal recording
python inference.py --input_dir demo_images/ --output results.json --debug

# Show output:
- Clean image: 0.92 confidence (AI)
- Same image compressed: 0.89 confidence (still catches it!)
- Real photo: 0.08 confidence (correctly real)
```
- Show results JSON
- Highlight: "Maintained accuracy despite transforms"

**6. Results (30 seconds)**
- Table comparing:
  - Frequency only: 88% → 72% (heavy transform)
  - Spatial only: 94% → 80% (heavy transform)
  - Our fusion: 96% → 87% (maintained!)
- "7% better than individual branches under stress"

**7. Innovation Highlight (15 seconds)**
- "Transform-aware fusion is our key innovation"
- "Not just an ensemble - adaptive intelligence"
- Show fusion weights changing based on image condition

**8. Impact & Call to Action (10 seconds)**
- "Real-world ready: handles compressed, edited, reposted images"
- "GitHub: [link]"
- "Try it yourself!"

### Presentation Talking Points

**For Judges Q&A:**

1. **Technical Execution:**
   - "We chose ConvNeXt-Tiny for efficiency while staying well under 2B params"
   - "Modular design: each branch independently testable"
   - "Reproducible: complete training scripts, seed control"

2. **Innovation:**
   - "Key insight: different detectors have complementary robustness profiles"
   - "Transform-aware fusion learns when to trust each branch"
   - "Not a naive ensemble - intelligent collaboration"

3. **Impact:**
   - "Addresses real-world scenario: images get compressed, edited, reposted"
   - "87% accuracy under heavy transforms vs 80% for single models"
   - "Production-ready architecture: fast inference, scalable"

4. **Problem Understanding:**
   - "We recognized the competition isn't just about accuracy - it's about robustness"
   - "Trained specifically on competition transforms"
   - "Extensive robustness testing validated our approach"

5. **Trade-offs & Honesty:**
   - "We chose 2 branches over 5 for simplicity and training time"
   - "ConvNeXt-Tiny over larger models: speed vs marginal accuracy gain"
   - "With more time: add PRNU analysis, test adversarial robustness"

### GitHub Repository

**README Structure:**
```markdown
# AI-Generated Image Detection - Transform-Aware Ensemble

## Overview
[Problem description, our approach, key innovation]

## Architecture
[Diagram + component descriptions]

## Installation
pip install -r requirements.txt

## Quick Start
python inference.py --input_dir images/ --output results.json

## Training from Scratch
[Step-by-step instructions]

## Results
[Metrics table, robustness comparison]

## Demo
[Link to YouTube video]

## Team Contributions
- Person A: Spatial detector
- Person B: Fusion model
- ...

## Limitations & Future Work
[Honest discussion]

## Citation
[If applicable]
```

**Code Quality:**
- Clear comments explaining non-obvious logic
- Type hints where helpful
- Consistent style (PEP 8)
- Docstrings for public functions

## Error Analysis Framework

### False Positive Analysis

**Categories to Investigate:**
1. **Over-processed Real Photos:**
   - Heavy Instagram filters
   - HDR photography
   - Aggressive sharpening
   - **Why it fools us:** Artifacts mimic AI generation

2. **Professional Edited Photos:**
   - Composites (stitched images)
   - Heavy Photoshop work
   - **Why it fools us:** Manual editing introduces similar artifacts

3. **Specific Image Types:**
   - Portraits with smooth skin (beauty filters)
   - Landscapes with unnatural colors
   - **Why it fools us:** Resembles AI aesthetic

**Mitigation Strategies:**
- If >5% false positives are over-processed photos: add them to training
- Document: "System may flag heavily edited photos"

### False Negative Analysis

**Categories to Investigate:**
1. **Specific Generator Types:**
   - Which AI model generates images we miss?
   - Stable Diffusion vs DALL-E vs Midjourney differences
   - **Action:** If one generator dominates failures, add more training data

2. **Image Content:**
   - Simple scenes (easier for AI to render perfectly)
   - Photorealistic portraits
   - **Insight:** More realistic = harder to detect

3. **Transform Combinations:**
   - Which combo breaks us most?
   - JPEG30 + Blur2.0 + Noise0.10?
   - **Action:** If specific combo fails, increase that augmentation during training

**Mitigation Strategies:**
- Analyze top 20 false negatives
- Find patterns (content, generator, transforms)
- Retrain with focused augmentation

### Representative Examples

**Collect for Presentation:**
- 5 hardest correct detections (low confidence but correct)
- 5 worst false positives (why did we miss?)
- 5 worst false negatives (why did we miss?)
- Show judges we understand our weaknesses

## Trade-offs & Design Decisions

### Decision Log

| Decision | Alternatives Considered | Rationale |
|----------|-------------------------|-----------|
| ConvNeXt-Tiny | EfficientNet-B0, ResNet-50, ViT-Small | Best balance of speed and accuracy for local artifact detection |
| 2 branches | 3+ branches (add PRNU, ELA, etc.) | Time constraint + diminishing returns + complexity |
| DCT over FFT | Fourier Transform | JPEG operates in DCT space, better robustness |
| 70/30 clean/aug | 50/50 or 80/20 | Prevents overfitting to degraded images while building robustness |
| Meta-fusion MLP | Attention-based fusion, weighted average | MLP sufficient, faster to train, more explainable |
| PyTorch | TensorFlow, JAX | Team familiarity, better timm integration |

### Scope Decisions

**In Scope:**
- Image-level detection (single image → confidence score)
- Robustness to listed transforms
- Explainability (branch contributions)

**Out of Scope:**
- Video detection
- Real-time streaming inference
- Adversarial robustness (attack/defense)
- Production deployment infrastructure
- Non-image modalities (audio, text)

**Why:**
- 3-day timeline → focus on core problem
- Competition requirements → stay aligned
- Proof of concept → not production system

## Limitations & Future Work

### Known Limitations

1. **Novel Generators:**
   - May not generalize to future AI models
   - Training distribution bias

2. **Extreme Transforms:**
   - Heavy compound transforms degrade performance
   - No silver bullet for heavily degraded images

3. **Edge Cases:**
   - Very small images (<100px)
   - Black & white images (color features break)
   - Extreme aspect ratios

4. **Adversarial Robustness:**
   - Not trained against evasion attacks
   - Could be fooled by adversarial examples

### Future Improvements

**With More Time:**
1. **Additional Branches:**
   - PRNU (Photo Response Non-Uniformity) analysis
   - ELA (Error Level Analysis)
   - Attention-based localization

2. **Model Improvements:**
   - Ensemble multiple architectures in Branch 2
   - Transformer-based fusion
   - Test-time augmentation

3. **Robustness Enhancements:**
   - Adversarial training
   - Self-supervised pretraining on domain data
   - Meta-learning for fast adaptation

4. **Explainability:**
   - Grad-CAM heatmaps (what regions triggered detection)
   - Frequency visualization (which frequencies matter)
   - Per-branch confidence calibration

5. **Production Features:**
   - API server (REST/gRPC)
   - Batch processing pipeline
   - Model versioning and A/B testing
   - Monitoring and drift detection

## Success Metrics

### Competition Performance

**Target Rankings:**
- Top 3: Strong showing, likely prize
- Top 5: Very competitive
- Top 10: Solid performance

**Key Differentiators vs Competitors:**
1. **Technical:** Adaptive fusion (not naive ensemble)
2. **Innovation:** Transform-aware decision making
3. **Presentation:** Clear problem understanding + strong demo
4. **Completeness:** Working code + reproducible results + honest error analysis

### Internal Validation

**Before Submission Checklist:**
- [ ] All three branches trained and validated
- [ ] Fusion model outperforms individual branches by ≥3%
- [ ] Robustness testing complete on all transform types
- [ ] Inference script produces correct JSON format
- [ ] Code runs on fresh environment (test reproducibility)
- [ ] Demo video recorded and uploaded
- [ ] README complete with setup instructions
- [ ] Error analysis document written
- [ ] All code commented and cleaned
- [ ] GitHub repository public

## Conclusion

This design proposes a **Transform-Aware Multi-Branch Ensemble** that intelligently combines frequency analysis and deep spatial features based on detected image characteristics. The approach directly addresses the competition's core challenge of robustness under real-world transformations.

**Key Strengths:**
- **Technical:** Modular, testable, reproducible
- **Innovative:** Adaptive fusion (not naive ensemble)
- **Practical:** Production-ready architecture, fast inference
- **Feasible:** Achievable in 3 days with 5-person team
- **Compelling:** Strong narrative for presentation

**Expected Outcome:** Top 3 finish with high scores in technical execution, innovation, and impact categories.

---

**Next Steps:** Review this design, then proceed to implementation planning with the `writing-plans` skill.
