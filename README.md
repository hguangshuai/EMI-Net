
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

## 📋 Requirements

### Software Dependencies

**Python Version:** 3.8+ (tested on Python 3.8, 3.9, 3.10, 3.11)

**Core Dependencies:**
- `torch==2.2.2` - PyTorch deep learning framework
- `torchvision==0.17.2` - Computer vision utilities
- `torchaudio==2.2.2` - Audio processing utilities
- `numpy==1.26.3` - Numerical computing
- `pandas==2.2.3` - Data manipulation and analysis
- `scikit-learn==1.6.1` - Machine learning utilities
- `h5py==3.13.0` - HDF5 file format support
- `matplotlib==3.9.4` - Plotting and visualization
- `SALib==1.5.1` - Sensitivity analysis library
- `pytorch-ignite==0.5.2` - Training utilities

**Additional Dependencies:**
- `scipy==1.13.1` - Scientific computing
- `pillow==11.0.0` - Image processing
- `contourpy==1.3.0` - Contour plotting
- `dill==0.3.9` - Serialization
- `joblib==1.4.2` - Parallel computing
- `networkx==3.2.1` - Graph algorithms

### Operating Systems

**Tested on:**
- macOS 13.0+ (Apple Silicon M1/M2 and Intel)
- Ubuntu 20.04 LTS, 22.04 LTS
- Windows 10/11 (with WSL2 recommended)

### Hardware Requirements

**Minimum Requirements:**
- CPU: Intel i5/AMD Ryzen 5 or equivalent
- RAM: 8GB
- Storage: 2GB free space
- GPU: Optional (CUDA-compatible GPU recommended for faster training)

**Recommended Requirements:**
- CPU: Intel i7/AMD Ryzen 7 or equivalent
- RAM: 16GB+
- Storage: 5GB free space
- GPU: NVIDIA GPU with 4GB+ VRAM (GTX 1060 or better)

**Tested Hardware:**
- NVIDIA RTX 2080, L40S
- Apple M1 Pro, M2 Max
- Intel i7-10700K, AMD Ryzen 7 5800X

---

## 🚀 Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/EMI-net.git
cd EMI-net
```

### Step 2: Create Virtual Environment

**Using venv (recommended):**
```bash
python -m venv venv
source venv/bin/activate   # For Linux/macOS
# OR
venv\Scripts\activate      # For Windows
```

**Using conda:**
```bash
conda create -n emi-net python=3.9
conda activate emi-net
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Typical Installation Time:**
- **Fast internet connection (100+ Mbps):** 5-10 minutes
- **Standard internet connection (25-50 Mbps):** 15-25 minutes
- **Slow internet connection (<10 Mbps):** 30-45 minutes

**Installation includes:**
- PyTorch with CUDA support (if CUDA is available)
- All scientific computing libraries
- Visualization tools
- Sensitivity analysis libraries

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🎯 Demo

### Quick Start Demo

The repository includes a sample dataset (`Data/CAI_data.h5`) for immediate testing.

#### Step 1: Run Training Demo

```bash
python train_model.py
```

**Expected Output:**
```
Original feature shape: (X, Y, Z)
Flattened shape: (X, Y)
Training model with hyperparameters: {...}
Epoch 1/100, Loss: X.XXX
...
Epoch 100/100, Loss: X.XXX
Test MAE: X.XX
Model saved as: best_model.pth
Results saved as: train_results.csv, test_results.csv
```

**Expected Runtime:**
- **CPU only:** 10-20 minutes
- **With GPU:** 3-8 minutes

#### Step 2: Run Sensitivity Analysis Demo

```bash
python sobol_sensitivity_analysis.py
```

**Expected Output:**
```
Loading pretrained model...
Running Sobol sensitivity analysis...
S1 indices: [array of sensitivity values]
ST indices: [array of total sensitivity values]
Sensitivity analysis completed.
Results saved as: sobol_results.csv
```

**Expected Runtime:**
- **CPU only:** 10-20 minutes
- **With GPU:** 2-5 minutes

### Demo Output Files

After running the demo, you'll find:

1. **`best_model.pth`** - Trained PyTorch model
2. **`train_results.csv`** - Training predictions and actual values
3. **`test_results.csv`** - Test set predictions and actual values
4. **`sobol_results.csv`** - Sensitivity analysis results

### Expected Performance

**Typical Model Performance:**
- **Test MAE:** 1.5-4.5 MPa

---

## 📖 Instructions for Use

### Running on Your Own Data

#### Step 1: Prepare Your Data

1. **Data Format:** Your data should be in HDF5 (`.h5`) format
2. **File Structure:** Place your `.h5` files in the `Data/` folder
3. **Data Requirements:**
   - EMI frequency response data
   - Compressive strength labels
   - Temperature and age information (optional)

#### Step 2: Modify Data Loading

Edit the file paths in your training script:

```python
# In train_model.py or sobol_sensitivity_analysis.py
file_list = ['your_data_1.h5', 'your_data_2.h5', 'your_data_3.h5']
file_list = [direct_list + i for i in file_list]
```

#### Step 3: Adjust Parameters

**For Different Frequency Ranges:**
```python
Raw_data = RawLoader(file_list, start_freq=10, end_freq=500)  # Adjust frequency range
```

**For Different Model Architectures:**
```python
# Modify hyperparameters in train_model.py
hyperparameters_space = {
    'learning_rate': [0.01, 0.001],  # Try different learning rates
    'batch_size': [32, 64, 128],     # Adjust batch size
    'conv1_out_channels': [32, 64, 128],  # Modify network architecture
    # ... other parameters
}
```

#### Step 4: Run Training

```bash
# For train/test split
python train_model.py

# For full dataset training
python train_no_split.py
```

#### Step 5: Analyze Results

```bash
# Run sensitivity analysis
python sobol_sensitivity_analysis.py

# Visualize results (optional)
python -c "
import pandas as pd
import matplotlib.pyplot as plt
results = pd.read_csv('train_results.csv')
plt.scatter(results['Actual'], results['Predicted'])
plt.xlabel('Actual Strength (MPa)')
plt.ylabel('Predicted Strength (MPa)')
plt.show()
"
```

### Customization Options

#### Model Architecture
- Modify `CnnRegressor` class in `Model_utli.py`
- Adjust number of convolutional layers
- Change activation functions
- Modify loss functions

#### Data Preprocessing
- Adjust frequency range in `RawLoader`
- Modify feature extraction methods
- Add data augmentation techniques

#### Training Parameters
- Change number of epochs
- Adjust learning rate schedules
- Modify optimizer settings
- Add early stopping

### Troubleshooting

**Common Issues:**

1. **CUDA Out of Memory:**
   ```bash
   # Reduce batch size in hyperparameters
   'batch_size': [16, 32]  # Instead of [64, 128]
   ```

2. **Data Loading Errors:**
   - Ensure `.h5` files are in correct format
   - Check file paths in scripts
   - Verify data structure matches expected format

3. **Installation Issues:**
   ```bash
   # Reinstall PyTorch with correct CUDA version
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

---

## 📁 Project Structure

```
EMI-net/
│
├── Model_utli.py              # Feature extraction, model definitions, loss function
├── train_model.py             # Model training with train/test split
├── train_no_split.py          # Training using the entire dataset (for final models)
├── sobol_sensitivity_analysis.py # Sobol sensitivity analysis on trained models
├── requirements.txt           # Python dependencies with exact versions
├── Data/                      # Folder containing .h5 data files
│   └── CAI_data.h5           # Sample dataset
└── README.md                  # This file
```

---

## 📊 Example Outputs

- **Prediction vs Time**
  - From `train_results.csv` after model training
- **Sobol Matrix**
  - Interpret feature interactions driving model prediction
- **Model Performance Metrics**
  - MAE, R², and loss curves

---

## ⚠️ Important Notes

### CPU vs GPU Performance
Model training results may slightly differ between CPU and GPU due to differences in floating point computation precision and parallelization behavior. These variations are typically minor.

**Performance Note:** When using all data with LS40S GPU for training, the test set MAE is around 1.91.

### Data Requirements
- Ensure your EMI data covers the frequency range specified in the scripts
- Compressive strength labels should be in MPa
- Data should be properly normalized/preprocessed

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

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
