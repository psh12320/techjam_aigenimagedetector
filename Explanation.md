# AI-Generated Image Detection System

## 🎯 The Problem

**Competition:** TechJam - AI-Generated Image Detection Challenge  
**Goal:** Build a system that can detect AI-generated images vs. real photographs  
**Deadline:** September 1, 2026

### The Challenge

Modern AI image generators (Stable Diffusion, Midjourney, DALL-E) create incredibly realistic images that are hard to distinguish from real photos. As these tools become more accessible, detecting AI-generated content becomes critical for:

- **Combating misinformation** (fake news with AI-generated images)
- **Protecting intellectual property** (detecting AI-generated art)
- **Maintaining content authenticity** (journalism, legal evidence)

### The Hard Part: Real-World Robustness

Images online don't stay pristine. They get:
- **Compressed** (JPEG artifacts from social media)
- **Blurred** (low-quality cameras, motion)
- **Noisy** (sensor noise, low light)
- **Resized** (thumbnails, responsive design)
- **Color-adjusted** (filters, editing)
- **Cropped** (framing, aspect ratios)

Most AI detectors fail when images are transformed. Our system must work even after these real-world degradations.

---

## 💡 Our Solution: Transform-Aware Multi-Branch Ensemble

Instead of a single model, we built **three specialized detectors** that each look at images differently:

### 1. Frequency Detector (DCT-Based)

**What it does:** Analyzes images in the frequency domain using Discrete Cosine Transform (DCT).

**Why it works:**
- AI generators produce images with **different frequency patterns** than real cameras
- Real photos have sensor noise and lens artifacts that show up in frequency space
- AI-generated images have suspiciously "perfect" frequency distributions

**Architecture:**
- Extracts 64-dimensional DCT features from 8×8 blocks
- Simple MLP: 64 → 128 → 64 → 1
- **Robust to compression** (JPEG also uses DCT)

**Training:**
- 20 epochs
- Light augmentation (20% color jitter/crop only)
- Learns frequency "fingerprints" of AI generation

---

### 2. Spatial Detector (ConvNeXt-Tiny CNN)

**What it does:** Analyzes pixel patterns and textures directly.

**Why it works:**
- AI generators struggle with fine details (hair, skin texture, text)
- ConvNeXt-Tiny is a modern CNN pre-trained on ImageNet
- Learns to spot artifacts invisible to humans (edge inconsistencies, unnatural smoothness)

**Architecture:**
- ConvNeXt-Tiny backbone (28M parameters)
- Custom classifier: 768 → 256 → 1
- Transfer learning from ImageNet

**Training:**
- **Phase 1 (5 epochs):** Freeze backbone, train classifier only
- **Phase 2 (20 epochs):** Fine-tune entire network
- Heavy augmentation (30% of images get random transforms)
- **Learns robustness** by training on degraded images

---

### 3. Fusion Model (Adaptive Meta-Learner)

**What it does:** Combines both detectors' predictions intelligently.

**Why it works:**
- Different detectors are better under different conditions
- **Clean images:** Spatial detector is more accurate
- **Compressed images:** Frequency detector is more reliable
- Fusion learns *when to trust which detector*

**Architecture:**
- Input: [freq_score, spatial_score, compression, blur, noise, color_shift, resolution] (7 dimensions)
- MLP: 7 → 64 → 32 → 16 → 1
- **Transform-aware:** Adapts based on detected image quality

**Training:**
- Generates predictions from both detectors on validation set
- Learns optimal weighting for each scenario
- 15 epochs with early stopping

---

## 🧠 Why This Approach Works

### 1. Complementary Detection
- **Frequency detector** catches global generation patterns
- **Spatial detector** catches local texture artifacts
- **Together:** Cover blind spots of each other

### 2. Transform Awareness
- **Problem:** Most detectors degrade on compressed/noisy images
- **Our solution:** 
  - Train spatial detector on augmented data (learns robustness)
  - Frequency detector naturally robust to compression
  - Fusion adjusts confidence based on detected transforms

### 3. Ensemble Strategy
- Single models overfit to training data
- Ensemble reduces variance and improves generalization
- Meta-learning finds optimal combination

---

## 📊 Implementation Details

### Dataset

**CIFAKE Dataset:**
- 10,000 real images (cameras, stock photos)
- 10,000 AI-generated images (StyleGAN, Stable Diffusion)
- Split: 80% train / 10% validation / 10% test

**Optional (if disk space allows):**
- SID_Set: 80,000+ images from diverse generators
- WildFake: 100,000+ wild-collected examples

### Data Augmentation Strategy

**RobustAugmentation:**
```python
# Spatial Detector (30% augmentation)
- JPEG compression: quality [30, 50, 70, 90]
- Gaussian blur: σ [0.5, 1.0, 2.0]
- Gaussian noise: σ [0.02, 0.05, 0.10]
- Color jitter: ±20% brightness/saturation
- Center crop: 80%

# Frequency Detector (20% augmentation)
- Light color jitter and crop only
- Avoids heavy augmentation (distorts frequency features)
```

### Training Pipeline

**Phase 1: Frequency Detector (~2 hours)**
```bash
python training/train_frequency.py
```
- Fast training (simple MLP)
- Learns frequency-domain patterns
- Output: `checkpoints/frequency_detector_best.pth`

**Phase 2: Spatial Detector (~8 hours)**
```bash
python training/train_spatial.py
```
- Two-stage training (freeze then fine-tune)
- Most compute-intensive phase
- Output: `checkpoints/spatial_detector_best.pth`

**Phase 3: Fusion Model (~30 minutes)**
```bash
python training/train_fusion.py
```
- Generates predictions from both detectors
- Trains meta-learner on combined features
- Output: `checkpoints/fusion_model_best.pth`

### Model Parameters

**Total Model Size:** ~35M parameters (well under 2B requirement)

| Component | Parameters | Purpose |
|-----------|-----------|---------|
| Frequency Detector | ~16K | Fast frequency analysis |
| Spatial Detector | ~28M | Deep texture analysis |
| Fusion Model | ~3K | Adaptive combination |
| Image Analyzer | 0 (rule-based) | Transform detection |

---

## 🚀 How to Run This Model (Any Computer)

### Step 1: Download Trained Models

If training on Google Colab, download the package:

```python
# In Colab (after training completes)
!zip -r trained_models.zip checkpoints/ inference.py models/ data/
from google.colab import files
files.download('trained_models.zip')
```

### Step 2: Set Up New Computer

**Requirements:**
- Python 3.8+
- No GPU needed (CPU works, just slower)

**Install dependencies:**
```bash
pip install torch torchvision timm opencv-python scikit-image numpy
```

**Extract models:**
```bash
unzip trained_models.zip
cd trained_models/
```

### Step 3: Run Inference

**Basic usage:**
```bash
python inference.py --input_dir test_images/ --output results.json
```

**Debug mode (see individual detector scores):**
```bash
python inference.py --input_dir test_images/ --output results.json --debug
```

**Competition submission:**
```bash
python inference.py --input_dir competition_dataset/ --output submission.json
```

### Step 4: Interpret Results

**Output format (JSON):**
```json
[
  {
    "image_path": "photo1.jpg",
    "pred": 0.05
  },
  {
    "image_path": "ai_image.jpg",
    "pred": 0.92
  }
]
```

**Interpretation:**
- **pred ≥ 0.5:** AI-generated (fake)
- **pred < 0.5:** Real photograph
- **pred close to 0.0:** Very confident it's real
- **pred close to 1.0:** Very confident it's AI-generated
- **pred around 0.5:** Uncertain (borderline case)

---

## 🔬 Key Technical Decisions

### Why ConvNeXt-Tiny Instead of ResNet?

**ConvNeXt advantages:**
- Modern architecture (2022) with Vision Transformer principles
- Better feature representations than ResNet
- More parameter-efficient
- Pre-trained on ImageNet (better transfer learning)

### Why DCT Instead of Raw Frequency?

**DCT advantages:**
- Same transform used in JPEG compression
- Model learns JPEG-robust features naturally
- Computationally efficient (fast inference)
- Well-studied in image forensics

### Why Two-Phase Training for Spatial Detector?

**Rationale:**
- Phase 1: Adapt classifier to our task (fast, prevents catastrophic forgetting)
- Phase 2: Fine-tune backbone for domain-specific features
- Prevents overfitting by gradual adaptation

### Why Separate Augmentation Strategies?

**Frequency detector:**
- Heavy augmentation distorts DCT features
- Light augmentation preserves frequency patterns

**Spatial detector:**
- Heavy augmentation teaches robustness
- Learns to detect AI regardless of transforms

### Why Meta-Learning for Fusion?

**Alternative:** Fixed averaging (freq_score + spatial_score) / 2  
**Problem:** Doesn't adapt to image quality

**Our approach:**
- Learns optimal weighting per scenario
- Compressed image → trust frequency more
- Clean image → trust spatial more
- **Result:** Better accuracy across all conditions

---

## 📈 Expected Performance

**On clean images:**
- Frequency Detector: ~62% accuracy
- Spatial Detector: ~85-90% accuracy
- **Fusion Model: ~90-95% accuracy**

**On transformed images:**
- JPEG compressed: ~85% accuracy
- Gaussian blurred: ~80% accuracy
- Noisy: ~82% accuracy
- Resized: ~88% accuracy
- **Average across transforms: ~84% accuracy**

**Why this is good:**
- Most single-model detectors drop to 60-70% on transforms
- Our ensemble maintains high accuracy across scenarios
- Transform-aware fusion adapts to degradation

---

## 🛠️ Project Structure

```
techjam_aigenimagedetector/
├── data/
│   ├── transforms.py          # Augmentation functions
│   ├── dataset.py             # Dataset loader
│   └── processed/
│       ├── real/              # Real images
│       └── fake/              # AI-generated images
├── models/
│   ├── frequency_detector.py  # DCT-based detector
│   ├── spatial_detector.py    # ConvNeXt-Tiny detector
│   ├── fusion_model.py        # Meta-learner
│   └── image_analyzer.py      # Transform detector
├── training/
│   ├── utils.py               # Training utilities
│   ├── train_frequency.py     # Phase 1 training
│   ├── train_spatial.py       # Phase 2 training
│   └── train_fusion.py        # Phase 3 training
├── evaluation/
│   ├── evaluate.py            # Test set evaluation
│   └── robustness_test.py     # Transform robustness test
├── checkpoints/               # Saved models (created during training)
├── inference.py               # Prediction script
├── train_on_colab.ipynb       # Google Colab training notebook
└── README.md                  # Quick start guide
```

---

## 🎓 What Makes This Approach Novel?

### 1. Transform-Aware Fusion
- Most ensembles use fixed weighting
- We adapt based on detected image transformations
- **Innovation:** Image quality guides detector trust

### 2. Frequency + Spatial Complementarity
- Frequency domain: robust to compression, catches global patterns
- Spatial domain: catches local artifacts, better on clean images
- **Innovation:** Combine orthogonal detection strategies

### 3. Robust Training Strategy
- Train spatial detector on degraded images
- Train frequency detector on clean images
- **Innovation:** Each detector optimized for its strength

### 4. Lightweight Architecture
- Only 35M parameters total
- Real-time inference (< 100ms per image on GPU)
- **Innovation:** Efficiency without sacrificing accuracy

---

## 🏆 Competition Strategy

### Why We'll Win

**1. Robustness**
- Competition tests on transformed images
- Our system explicitly trained for this
- Other teams likely using single models (brittle)

**2. Ensemble Advantage**
- Reduces overfitting to training distribution
- Covers more failure modes
- More reliable than any single detector

**3. Smart Fusion**
- Adapts to image quality automatically
- No manual threshold tuning needed
- Generalizes to unseen transforms

**4. Training Data Diversity**
- CIFAKE covers multiple generators
- Optional SID_Set/WildFake for more diversity
- Broad training = better generalization

---

## 🔮 Future Improvements

**If we had more time/compute:**

1. **Larger models:** ConvNeXt-Base or ViT-Large (100M+ params)
2. **More datasets:** Train on all 3 datasets (~300K images)
3. **Test-time augmentation:** Predict on 5 augmented versions, average
4. **Attention visualization:** Show which regions look fake
5. **Generator-specific detectors:** Separate heads for Stable Diffusion, Midjourney, etc.
6. **Self-supervised pre-training:** Learn from unlabeled images first

**Current constraints:**
- Limited Colab GPU time (12 hours)
- Small dataset (20K images)
- Parameter budget (< 2B)
- Deadline (September 1)

**Our approach optimizes for these constraints** while maximizing performance.

---

## 📝 Key Takeaways

**What we built:**
- Transform-aware multi-branch ensemble for AI image detection
- Three specialized detectors (frequency, spatial, fusion)
- Robust to real-world image transformations
- Lightweight (35M parameters) and fast inference

**Why it works:**
- Complementary detection strategies cover different failure modes
- Meta-learning adapts to image quality
- Robust training strategy teaches generalization
- Ensemble reduces overfitting and improves reliability

**How to use it:**
- Download trained models from Colab
- Run `inference.py` on any computer (no GPU required)
- Get JSON predictions for competition submission
- Interpretable scores (0.0 = real, 1.0 = fake)

**The innovation:**
- Not just another CNN detector
- First transform-aware adaptive ensemble
- Optimized for real-world robustness
- Practical and deployable

---

## 🙏 Credits

**Competition:** TechJam - AI-Generated Image Detection Challenge  
**Dataset:** CIFAKE (real + AI-generated images)  
**Framework:** PyTorch 2.0+  
**Pre-trained models:** timm (PyTorch Image Models)  
**Training platform:** Google Colab (free T4 GPU)  

**Built for winning 1st place.** 🏆

---

## 📧 Contact

Questions about the implementation? Check:
- `README.md` - Quick start guide
- `train_on_colab.ipynb` - Training walkthrough
- `inference.py` - Usage examples

**Let's detect those fakes!** 🔍✨
