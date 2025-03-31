# =============================================================================
# Guangshuai Han (Jerry) @ Purdue University, Luna Group
# Full-Data Training Script (No Train/Test Split)
# =============================================================================

from Model_utli import CnnRegressor, RawLoader, model_loss
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD, Adam
import numpy as np
from sklearn.model_selection import train_test_split
import itertools
import pandas as pd

# -----------------------------
# Load Data
# -----------------------------
direct_list = 'Data/'
file_list = ['slab_1.h5', 'slab_2.h5', 'slab_3.h5']
file_list = [direct_list + i for i in file_list]

Raw_data = RawLoader(file_list, start_freq=10, end_freq=500)
Data_x = np.array(Raw_data.fea_spec_temp_logage())  # shape: [samples, freq, 2]
Data_y = np.array(Raw_data.Label)                   # shape: [samples, ]

# -----------------------------
# Device Setup
# -----------------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# -----------------------------
# Hyperparameter Space (Single or Grid)
# -----------------------------
# You can expand this space for grid search
hyperparameters_space = {
    'learning_rate': [0.001],
    'optimizer': ['Adam'],
    'batch_size': [64],
    'conv1_out_channels': [64],
    'conv2_out_channels': [64],
    'linear1_out_features': [64],
    'linear2_out_features': [32]
}

# =============================================================================
# Function: train_and_evaluate_model
# Description: Train model using all data without split. Return train loss.
# =============================================================================
def train_and_evaluate_model(hyperparams):
    epochs = 150
    batch_size = hyperparams['batch_size']

    # Use all data for training
    train_dataset = TensorDataset(
        torch.from_numpy(Data_x).float(),
        torch.from_numpy(Data_y).reshape(-1, 1).float()
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Build model
    model = CnnRegressor(Data_x.shape[1], 1,
                         hyperparams['conv1_out_channels'],
                         hyperparams['conv2_out_channels'],
                         hyperparams['linear1_out_features'],
                         hyperparams['linear2_out_features']).to(device)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=hyperparams['learning_rate']) \
        if hyperparams['optimizer'] == 'Adam' \
        else SGD(model.parameters(), lr=hyperparams['learning_rate'])

    # Training loop
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.MSELoss()(outputs, targets)
            loss.backward()
            optimizer.step()

    # Evaluate training loss after training
    model.eval()
    with torch.no_grad():
        train_losses = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            train_loss = torch.nn.MSELoss()(outputs, targets)
            train_losses.append(train_loss.item())

    avg_train_loss = np.mean(train_losses)
    return model, avg_train_loss, train_loader

# =============================================================================
# Function: save_model_and_results
# Description: Save model and prediction results on training set
# =============================================================================
def save_model_and_results(model, train_loader, hyperparams,
                           model_name="best_model.pth",
                           train_result_csv="train_results.csv"):
    model.eval()
    torch.save(model.state_dict(), model_name)

    predictions, actuals, times = [], [], []

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        time_data = inputs[:, -1, 0].cpu().numpy()  # log(age) position
        with torch.no_grad():
            output = model(inputs)
        predictions.extend(output.view(-1).cpu().numpy())
        actuals.extend(targets.view(-1).cpu().numpy())
        times.extend(time_data)

    # Convert log(age) back to age
    times = np.exp(times)

    df_results = pd.DataFrame({
        "Time": times,
        "Actual": actuals,
        "Predicted": predictions
    })
    df_results.to_csv(train_result_csv, index=False)

# =============================================================================
# Main Hyperparameter Search Loop
# =============================================================================
best_loss = np.inf
best_hyperparams = None
best_model = None

for hyperparams in itertools.product(*[hyperparameters_space[key] for key in hyperparameters_space]):
    hyperparams_dict = dict(zip(hyperparameters_space.keys(), hyperparams))
    model, train_loss, train_loader = train_and_evaluate_model(hyperparams_dict)

    if train_loss < best_loss:
        best_loss = train_loss
        best_hyperparams = hyperparams_dict
        best_model_state = model.state_dict()
        print(f"New Best: Loss = {best_loss:.4f}, Hyperparams = {best_hyperparams}")

# =============================================================================
# Load & Save Final Best Model
# =============================================================================
if best_model_state is not None:
    best_model = CnnRegressor(Data_x.shape[1], 1,
                              best_hyperparams['conv1_out_channels'],
                              best_hyperparams['conv2_out_channels'],
                              best_hyperparams['linear1_out_features'],
                              best_hyperparams['linear2_out_features']).to(device)
    best_model.load_state_dict(best_model_state)
    print(f"✅ Final Best Hyperparameters: {best_hyperparams}, Final Loss: {best_loss:.4f}")
    save_model_and_results(best_model, train_loader, best_hyperparams)
