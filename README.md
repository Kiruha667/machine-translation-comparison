# Machine Translation: Seq2Seq vs Transformer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![GPU](https://img.shields.io/badge/GPU-GTX%201660%20SUPER-green.svg)](https://www.nvidia.com/en-us/geforce/graphics-cards/16-series/)

Comprehensive comparative study of Seq2Seq with Attention and Transformer models for English-French machine translation. Both models implemented from scratch in PyTorch.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Results](#results)
- [Analysis](#analysis)
- [GPU Configuration](#gpu-configuration)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [Contact](#contact)

## Overview

This project implements two state-of-the-art neural machine translation architectures:

### Seq2Seq with Attention
- **Encoder**: LSTM (embedding: 256, hidden: 512)
- **Attention**: Bahdanau attention mechanism
- **Decoder**: LSTM with context vector
- **Features**: Teacher forcing, greedy decoding

### Transformer
- **Architecture**: 4 encoder + 4 decoder layers
- **Attention**: 8-head multi-head self-attention
- **Dimension**: 512
- **Features**: Sinusoidal positional encoding

## Features

- From-scratch implementation - No pre-trained models
- GPU accelerated - Optimized for NVIDIA GTX 1660 SUPER
- Comprehensive evaluation - BLEU scores, translation samples, error analysis
- Visualizations - Training curves, performance comparisons
- Modular design - Easy to extend and modify
- Well documented - Clear code with extensive comments

## Project Structure

```
machine-translation-comparison/
│
├── README.md
├── requirements.txt
├── setup.py
│
├── data/
│   └── download_data.py
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── dataset.py
│   └── models/
│       ├── seq2seq.py
│       └── transformer.py
│
├── scripts/
│   ├── train_seq2seq.py
│   ├── train_transformer.py
│   └── compare_models.py
│
├── models/              # Saved checkpoints
├── outputs/            # Visualizations
└── logs/              # Training logs
```

## Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (GTX 1660 SUPER used)
- CUDA 11.0+
- 8GB RAM minimum

### Setup

```bash
# Clone repository
git clone https://github.com/Kiruha667/machine-translation-comparison.git
cd machine-translation-comparison

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Setup structure
python setup.py

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python check_gpu.py
```

## Quick Start

```bash
# 1. Download dataset
python data/download_data.py

# 2. Train Seq2Seq (~50 minutes)
python scripts/train_seq2seq.py --epochs 10 --batch-size 64

# 3. Train Transformer (~54 minutes)
python scripts/train_transformer.py --epochs 10 --batch-size 64

# 4. Compare models
python scripts/compare_models.py
```

## Results

### Performance Metrics

| Metric | Seq2Seq | Transformer | Winner |
|--------|---------|-------------|--------|
| **BLEU Score** | **45.50** | 28.19 | Seq2Seq |
| **Translation Quality** | Mixed | **Better** | Transformer |
| **Training Time** | **2931s** (~49min) | 3236s (~54min) | Seq2Seq |
| **Parameters** | **23M** | 56M | Seq2Seq |
| **GPU Memory** | **2-3GB** | 4-5GB | Seq2Seq |
| **Best Val Loss** | 3.286 | 3.286 | Tie |

### BLEU Score Paradox

**Important Finding:**
- Seq2Seq achieved **higher BLEU (45.50)** through shorter, conservative translations
- Transformer got **lower BLEU (28.19)** due to word repetitions being penalized
- However, **manual evaluation** shows Transformer produces better grammar
- **Lesson:** BLEU alone can be misleading - qualitative analysis is essential!

### Sample Translations Comparison

| # | Source (EN) | Seq2Seq (FR) | Transformer (FR) | Reference | Quality Winner |
|---|------------|--------------|------------------|-----------|----------------|
| 1 | i love you | vous que vous | je taime | je t'aime | Transformer |
| 2 | good morning | bonjour | bonjour × 4 | bonjour | Seq2Seq |
| 3 | how are you | comment êtesvous | comment vastu | comment allez-vous | Tie |
| 4 | thank you | je vous remercie* | merci × 3 | merci | Seq2Seq |
| 5 | see you tomorrow | à demain × 2 | on se voit demain × 3 | à demain | Seq2Seq |
| 6 | i am learning french | je le le français | japprends le français | j'apprends le français | Transformer |
| 7 | the weather is nice today | il fait beau aujourdhui × 2 | il fait beau aujourdhui | il fait beau aujourd'hui | Transformer |
| 8 | where is the station | où est trouve la gare | où se trouve la gare | où est la gare | Transformer |
| 9 | i dont understand | ne le comprends pas | je ne comprends pas × 2 | je ne comprends pas | Transformer |
| 10 | can you help me | pouvezvous maider | peuxtu maider | pouvez-vous m'aider | Tie |

**Qualitative Score: Transformer 5 wins | Seq2Seq 3 wins | 2 ties**

*Grammatically correct but overly formal

### Key Observations

**Seq2Seq Strengths:**
- Less word repetition
- Better at short, common phrases
- More stable output length
- Higher BLEU scores
- Faster training and less memory usage

**Seq2Seq Weaknesses:**
- Missing articles (le, la)
- Grammatical errors in complex sentences
- Sometimes nonsensical output ("vous que vous")

**Transformer Strengths:**
- Superior grammar and syntax
- Better on longer sentences
- Handles complex structures
- More natural translations

**Transformer Weaknesses:**
- Critical issue: Word repetition (2-4× per word)
- Doesn't know when to stop
- Over-generates for simple phrases
- Lower BLEU due to repetition penalties

**Common Issues (Both Models):**
- Missing apostrophes: `t'aime` → `taime`
- Missing hyphens: `allez-vous` → `êtesvous`
- Missing accents: `aujourd'hui` → `aujourdhui`

## Training Curves

![Training Curves](outputs/training_curves.png)
*Loss progression over 10 epochs for both models*

### Training Dynamics Analysis

**Seq2Seq:**
- Fast initial convergence (epochs 1-3)
- **Critical issue: Overfitting after epoch 4**
  - Training loss continues to decrease: 3.4 → 0.8
  - Validation loss increases: 3.3 → 3.6
- Model memorizes training data but generalizes poorly
- Early stopping at epoch 3-4 would be optimal

**Transformer:**
- Slower but stable convergence
- No overfitting observed
- Training and validation loss remain parallel
- Final losses: Train 0.5, Val 1.5
- Better generalization capability

**Key Insight:** Despite higher BLEU score, Seq2Seq's performance is artificially inflated due to overfitting on test set vocabulary. Transformer shows healthier learning dynamics.

![Model Comparison](outputs/model_comparison.png)
*BLEU and training time comparison*

### Understanding the BLEU Paradox

The graphs reveal why Seq2Seq has higher BLEU despite worse translation quality:

1. **Seq2Seq overfits**: Memorizes common phrase patterns from training data
2. **Test set similarity**: Test phrases are similar to training → high BLEU
3. **But poor generalization**: Novel phrases produce nonsense ("vous que vous")
4. **Transformer generalizes**: Lower BLEU but better grammar and structure

**Recommendation:** 
- For Seq2Seq: Add dropout (currently 0.1 → 0.3), reduce model capacity, implement early stopping
- For Transformer: Add repetition penalty during decoding
## Analysis

### 1. Effect of Attention in Seq2Seq
- Improves source-target alignment
- Handles long-range dependencies
- Reduces information bottleneck
- Provides interpretability

### 2. Self-Attention vs RNNs
- **Parallelization**: Entire sequence processed simultaneously
- **Gradient flow**: No vanishing gradient issues
- **Dependencies**: Better long-range modeling
- **Trade-off**: More parameters, higher memory

### 3. Multi-Head Attention Benefits
- Captures diverse linguistic phenomena
- Heads specialize in different patterns
- Increases model capacity
- Provides redundancy

### 4. Key Findings

**Quantitative (BLEU):**
- Seq2Seq wins (45.50 vs 28.19)
- Conservative translations match references better

**Qualitative (Manual):**
- Transformer produces better grammar
- Repetition problem significantly hurts usability
- Both need post-processing for punctuation

**Recommendations:**
1. Add repetition penalty during decoding
2. Implement better EOS detection
3. Post-process for apostrophes/accents
4. Use beam search with diversity penalty
5. Consider hybrid approach

## GPU Configuration

### GTX 1660 SUPER (6GB VRAM)

**Optimal Settings:**
```python
BATCH_SIZE = 64        # Reduce to 32 if OOM
NUM_WORKERS = 4
PIN_MEMORY = True
```

**Memory Usage:**
- Seq2Seq: ~2-3GB VRAM
- Transformer: ~4-5GB VRAM

**Monitor GPU:**
```bash
nvidia-smi -l 1
```

## Troubleshooting

### CUDA Not Available

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory

```bash
python scripts/train_transformer.py --batch-size 32
```

### Dataset Download Fails

Manual download: https://www.manythings.org/anki/fra-eng.zip
Extract to `data/fra.txt`

## References

1. **Bahdanau et al. (2014)** - Neural Machine Translation by Jointly Learning to Align and Translate
2. **Vaswani et al. (2017)** - Attention Is All You Need
3. **Sutskever et al. (2014)** - Sequence to Sequence Learning with Neural Networks
4. **Papineni et al. (2002)** - BLEU: Method for Automatic Evaluation of Machine Translation

## Contact

**Kirill Selivankin**
- Email: selivankink@gmail.com
- LinkedIn: [Kirill Selivankin](https://www.linkedin.com/in/кирилл-селиванкин-0220a1337)
- GitHub: [@Kiruha667](https://github.com/Kiruha667)


---

<div align="center">

**Machine Translation Research Project**

If this project was helpful, please consider starring the repository.

</div>
