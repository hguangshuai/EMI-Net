# =============================================================================
# Guangshuai Han (Jerry) @ Purdue University, Luna Group
# Sobol Sensitivity Analysis of CNN Predictions from EMI Signals
# =============================================================================

from Model_utli import CnnRegressor, RawLoader, model_loss
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD, Adam
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from SALib.sample import sobol as sobol_sampler
from SALib.analyze import sobol

# -----------------------------
# Task & Data Settings
# -----------------------------
task = '10-500_single'
direct_list = 'Data/'
file_list = ['slab_1.h5', 'slab_2.h5', 'slab_3.h5']
file_list = [direct_list + i for i in file_list]

# -----------------------------
# Load data and features
# -----------------------------
Raw_data = RawLoader(file_list, start_freq=10, end_freq=500)
Data_x = np.array(Raw_data.fea_spec_temp_logage())  # shape: [samples, freq, 2]
Data_y = np.array(Raw_data.Label)
batch_size = 32

print("Original feature shape:", Data_x.shape)

# -----------------------------
# Setup device
# -----------------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# -----------------------------
# Flatten EMI features: [n_samples, freq] * 2
# -----------------------------
Data_x_1 = Data_x[:, :, 0]
Data_x_2 = Data_x[:, :, 1]
Data_x = np.concatenate([Data_x_1, Data_x_2], axis=1)
print("Flattened shape:", Data_x.shape)

# -----------------------------
# Define parameter bounds for sampling
# -----------------------------
# ⚠️ IMPORTANT: Ensure bounds and feature names match the model input structure exactly.
# If you change the used features or model architecture (e.g., number of channels, input shape),
# you MUST update the `problem` definition (num_vars, names, bounds) accordingly.
sum_max = Data_x.max(axis=0)
sum_min = Data_x.min(axis=0)
bounds = [[sum_min[i], sum_max[i]] for i in range(sum_max.size)]
features_name = []

# Feature naming convention
for i in range(202):
    if i < 101:
        features_name.append(f'C{i+1}')
    else:
        features_name.append(f'B{i-101}')

print(f"Total Features: {len(features_name)}")
print("Bounds shape:", np.array(bounds).shape)

problem = {
    "num_vars": 202,
    "names": features_name,
    "bounds": bounds
}

# -----------------------------
# Sampling using Sobol method
# -----------------------------
param_values = np.array(sobol_sampler.sample(problem, 128))  # Recommended: >1000 for stability

# Reshape to 3D for model input: [samples, freq, channels]
P_1 = param_values[:, :101][:, :, np.newaxis]
P_2 = param_values[:, 101:][:, :, np.newaxis]
x_val = np.concatenate([P_1, P_2], axis=2)

# Dummy labels (not used for prediction)
y_val = np.zeros([x_val.shape[0], 1])

inputs = torch.from_numpy(x_val).float()
outputs = torch.from_numpy(y_val).float()
loader = DataLoader(TensorDataset(inputs, outputs), batch_size, shuffle=False, drop_last=True)

# -----------------------------
# Load pretrained model
# -----------------------------
# ⚠️ IMPORTANT: Ensure the model architecture matches the one used for training.
model = CnnRegressor(x_val.shape[1], 1,
                     64, 64, 64, 32).to(device)

pretrained_model_path = 'best_model.pth'
model.load_state_dict(torch.load(pretrained_model_path))
model.eval()

# -----------------------------
# Run inference on sampled points
# -----------------------------
pred = []
for input, _ in iter(loader):
    input = input.to(device)
    predictions = model(input).cpu().detach().numpy().ravel()
    pred.append(predictions)

Y = np.concatenate(pred)  # Final prediction vector

# -----------------------------
# Run Sobol Analysis
# -----------------------------
Si = sobol.analyze(problem, Y)

SA_data_1 = Si['S1']       # First-order indices
SA_data_2 = Si['S2']       # Second-order indices
SA_data_total = Si['ST']   # Total-order indices

# -----------------------------
# Save second-order matrix plot
# -----------------------------
plt.figure(figsize=(10, 8))
plt.imshow(SA_data_2, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Second Order Sobol Indices')
plt.xlabel('Feature Index')
plt.ylabel('Feature Index')
plt.xticks(range(len(features_name)), features_name, rotation=90)
plt.yticks(range(len(features_name)), features_name)
fig_matrix = task + "_second_order_matrix.jpg"
plt.tight_layout()
plt.savefig(fig_matrix)
plt.show()

# -----------------------------
# Save CSV of 2nd-order matrix
# -----------------------------
second_order_df = pd.DataFrame(SA_data_2, index=features_name, columns=features_name)
second_order_df.to_csv(task + '_second_order_matrix.csv')

# -----------------------------
# Save CSV of S1 / ST
# -----------------------------
first_order_df = pd.DataFrame(SA_data_1, index=features_name, columns=['First Order'])
total_order_df = pd.DataFrame(SA_data_total, index=features_name, columns=['Total Order'])
combined_df = pd.concat([first_order_df, total_order_df], axis=1)
combined_df.to_csv(task + '_sobol_indices.csv')

print("✅ Sobol analysis complete and saved.")
