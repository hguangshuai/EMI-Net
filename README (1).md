
# EMI-Net: AI-Driven Interpretation of Piezoelectric Responses in Concrete

**Author:** Guangshuai Han (Jerry) @ Purdue University, Luna Group  
**Project Goal:** Learn and predict early-age strength development in concrete using piezoelectric EMI signals and deep learning.

---

## 🔍 Overview

This repository implements a full pipeline for:

- 📥 Loading EMI signal data from `.h5` files
- 🧠 Training 1D CNN or simple NN models to predict strength from EMI signals
- 🔬 PCA-based feature visualization
- 🧪 Sobol sensitivity analysis for model interpretability
- 🧹 Optional data cleaning and preprocessing utilities

---

## 📁 Project Structure

```
EMI-net/
│
├── Model_utli.py              # Feature extraction, model definitions, loss function
├── train_model.py             # Model training with train/test split
├── train_no_split.py          # Training using the entire dataset (for final models)
├── sobol_sensitivity_analysis.py # Sobol sensitivity analysis on trained models
├── Data/                      # Folder containing slab_*.h5 files
└── README.md                  # This file
```

---

## 📊 Example Outputs

- **Prediction vs Time**
  - From `train_results.csv` after model training
- **Sobol Matrix**
  - Interpret feature interactions driving model prediction

---

## ⚙️ Setup Instructions

### ✅ Environment Setup (Recommended)

Use Python 3.8+ with virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows

pip install -r requirements.txt
```

---

## 🧪 Generate requirements.txt

You can auto-generate it from your virtual environment:

```bash
pip freeze > requirements.txt
```

Or create manually with the following base dependencies:

```text
numpy
pandas
torch>=1.12
h5py
scikit-learn
matplotlib
SALib
pytorch-ignite
```

---

## ⚠️ Note on CPU vs GPU

Model training results may slightly differ between CPU and GPU due to differences in floating point computation precision and parallelization behavior. These variations are typically minor.

---

## 🛠 Usage

### Step 1: Prepare Data
Put your `.h5` files (e.g. `slab_1.h5`, `slab_2.h5`...) into the `Data/` folder.

### Step 2: Train Model
```bash
python train_model.py      # with split
python train_no_split.py   # no split, full training
```

### Step 3: Evaluate & Analyze
```bash
python sobol_sensitivity_analysis.py
```

---

## 📌 Citation

This repository is associated with the manuscript:

> **"Sensing the Unsensed: AI Interprets Piezoelectric Whispers from Concrete"**  
> Guangshuai Han et al., Purdue University, 2024.  
> *(Currently under review)*

---

## 📬 Contact

If you have any questions, feel free to contact:  
📧 **hanguangshuai [at] gmail [dot] com**

---
