# HDCR: Cross-lingual Medical Misinformation Detection through Contrastive Claim-Evidence Reasoning

Official implementation of HDCR (Health Distortion Detector with Contrastive Reasoning), a framework for fine-grained detection of medical misinformation across languages.

## Overview

HDCR addresses the challenge of detecting subtle medical misinformation through:
- Fine-grained distortion taxonomy: 4 clinically-validated distortion types
- Contrastive reasoning: Explicit claim-evidence semantic comparison
- Cross-lingual capability: Robust performance across English and Chinese with only 2.9% degradation


## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 16GB+ RAM recommended


### Requirements

Create a requirements.txt file with:
```
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```
## Dataset

Our cross-lingual medical misinformation dataset is publicly available on Dryad:

Dataset Link: https://datadryad.org/stash/dataset/doi:10.5061/dryad.xxxxx


### Download and Setup

Create data directory:
mkdir -p data_splits

Download from Dryad:
Visit https://datadryad.org/stash/dataset/doi:10.5061/dryad.xxxxx
Download train.json, dev.json, test.json

Place files in data_splits/:
data_splits/train.json
data_splits/dev.json
data_splits/test.json

## Quick Start

### Step 1: Download Pre-trained Models

Create models directory:
```
mkdir -p models
```
Download XLM-RoBERTa and PubMedBERT (recommended):
```
huggingface-cli download xlm-roberta-base --local-dir models/xlm-roberta-base
huggingface-cli download microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract --local-dir models/pubmedbert-base
```

### Step 2: Train HDCR Model

Train with default configuration:
```
python train.py
```
Output will be saved to:
./hdcr_bge_pubmedbert_output/

### Step 3: Evaluate Model
```
# Evaluate single model:
python eva.py --input hdcr_bge_pubmedbert_output/test_predictions.json --visualize --language

# Compare multiple models:
python eva.py --compare model1_predictions.json model2_predictions.json model3_predictions.json --language

# Generate LaTeX table data for paper:
python eva.py --latex model1_predictions.json model2_predictions.json model3_predictions.json
```

## Evaluation Metrics

Binary Classification (Misinformation Detection):
- Precision, Recall, F1
- Matthews Correlation Coefficient (MCC)
- Balanced Accuracy

Multiclass Classification (Distortion Type Detection):
- Accuracy
- Macro F1
- Per-class Precision, Recall, F1

Cross-lingual Performance:
- English F1 scores
- Chinese F1 scores
- Cross-lingual degradation


## Project Structure
```
hdcr-medical-misinformation/
├── train.py                      # Main training script
├── eva.py         # Evaluation script
├── README.md                     # This file
├── data_splits/                  # Dataset directory
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── models/                       # Pre-trained models
│   ├── xlm-roberta-base/
│   ├── pubmedbert-base/
│   └── bge-m3/
└── hdcr_output/                  # Training outputs
    ├── best_model.pt
    └── test_predictions.json
```
## Citation

If you use this code or dataset in your research, please cite:


## License

This project is licensed under the MIT License.
