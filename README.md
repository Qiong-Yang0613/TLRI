# TLRI: A Derivatization-Agnostic Framework for Accurate Multi-Phase Retention Index Prediction

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the official implementation of **TLRI** (Transfer Learning-based Retention Index), a deep learning framework designed for high-precision gas chromatographic Kováts retention index (RI) prediction across multiple stationary phases.

## 🌟 Key Features

- **Derivatization-Agnostic**: Natively supports both native and derivatized (e.g., TMS, TBDMS) molecules without requiring explicit structural labeling.
- **Multi-Phase Support**: Capable of predicting RI for three common stationary phases: 
  - Semi-standard non-polar (**SSNP**)
  - Standard non-polar (**SNP**)
  - Standard polar (**SP**)
- **Hybrid Architecture**: Combines **AttentiveFP** (Graph Neural Network) with expert-engineered global descriptors (Morgan fingerprints, RDKit descriptors, and fragment-based descriptors).
- **Transfer Learning Strategy**: Leverages pre-training on a large-scale SSNP dataset (~120,000 entries) and fine-tuning on SNP/SP phases to overcome data scarcity.
- **Uncertainty & Interpretability**: Includes MC Dropout for predictive uncertainty quantification and Integrated Gradients (IG) for atom-level structural attribution.

## 🏗️ Model Architecture

The TLRI model fuses local structural motifs extracted by a GNN with macro-physicochemical descriptors. This hybrid representation is then processed through a Multi-Layer Perceptron (MLP) to output the final RI value.


## 📂 Repository Structure

- `train.py`: Script for model pre-training and phase-specific fine-tuning.
- `predict.py`: Script for inference using trained models.
- `Confidence_Interpretability.py`: Implementation of MC Dropout and Integrated Gradients for reliability and transparency.
- `data/`: Directory containing curated datasets (SSNP, SNP, SP).
- `extra test set information.csv`: The independent benchmark test set (n=500).
- `LICENSE`: Apache-2.0 license.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- PyTorch Geometric
- RDKit (v2023.03.2 or later)
- Scikit-learn, Pandas, Numpy

### Usage

#### 1. Prediction
To predict RI values for your molecules (SMILES format):
```bash
python predict.py --input your_molecules.csv --phase SNP
