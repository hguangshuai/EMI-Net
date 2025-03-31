# =============================================================================
# Guangshuai Han (Jerry) @ Purdue University, Luna Group
# CNN Model Training & Hyperparameter Search for EMI-Based Concrete Monitoring
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
# Data Preparation
# -----------------------------
direct_list = 'Data/'
file_list = ['slab_1.h5', 'slab_2.h5', 'slab_3.h5']
file_list = [direct_list + i for i in file_list]

Raw_data = RawLoader(file_list, start_freq=10, end_freq=200)
Data_x = np.array(Raw_data.fea_spec_temp_logage_NB())  # Features: RC only, with Temp + log(Age)
Data_y = np.array(Raw_data.Label)                      # Target: Compressive Strength

# -----------------------------
# Device Selection
# -----------------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# -----------------------------
# Hyperparameter Search Space
# -----------------------------
hyperparameters_space = {
    'learning_rate': [0.001, 0.0005, 0.0001],                 
    'optimizer': ['Adam', 'SGD'],                             
    'batch_size': [16, 32, 64],                              
    'conv1_out_channels': [32, 64, 128],                     
    'conv2_out_channels': [64, 128, 256],                    
    'linear1_out_features': [32, 64, 128],                    
    'linear2_out_features': [16, 32, 64]                     
}


# =============================================================================
# Function: train_and_evaluate_model
# Description: Train model with given hyperparams & return performance and loaders
# =============================================================================
def train_and_evaluate_model(hyperparams):
    epochs = 100
    batch_size = hyperparams['batch_size']

    # Split into training and test sets
    train_x, test_x, train_y, test_y = train_test_split(
        Data_x, Data_y, test_size=0.3, random_state=42
    )

    train_dataset = TensorDataset(
        torch.from_numpy(train_x).float(), torch.from_numpy(train_y).reshape(-1, 1).float()
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(
        torch.from_numpy(test_x).float(), torch.from_numpy(test_y).reshape(-1, 1).float()
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = CnnRegressor(Data_x.shape[1], 1,
                         hyperparams['conv1_out_channels'],
                         hyperparams['conv2_out_channels'],
                         hyperparams['linear1_out_features'],
                         hyperparams['linear2_out_features']).to(device)

    # Select optimizer
    optimizer = Adam(model.parameters(), lr=hyperparams['learning_rate']) \
        if hyperparams['optimizer'] == 'Adam' else \
        SGD(model.parameters(), lr=hyperparams['learning_rate'])

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

    # Evaluation on test set
    model.eval()
    with torch.no_grad():
        test_losses = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss = torch.nn.L1Loss()(outputs, targets)
            test_losses.append(test_loss.item())

    avg_test_loss = np.mean(test_losses)
    return model, avg_test_loss, train_loader, test_loader

# =============================================================================
# Function: save_model_and_results
# Description: Save trained model and predicted results to file
# =============================================================================
def save_model_and_results(model, train_loader, test_loader, hyperparams,
                           model_name="best_model.pth",
                           train_result_csv="train_results.csv",
                           test_result_csv="test_results.csv"):
    model.eval()
    torch.save(model.state_dict(), model_name)

    datasets = {'train': train_loader, 'test': test_loader}
    results_files = {'train': train_result_csv, 'test': test_result_csv}

    for key in datasets:
        times, predictions, actuals = [], [], []
        for inputs, targets in datasets[key]:
            inputs, targets = inputs.to(device), targets.to(device)
            time_data = inputs[:, -1, 0].cpu().numpy()  # Assumes log(age) stored here
            with torch.no_grad():
                output = model(inputs)
            predictions.extend(output.view(-1).cpu().numpy())
            actuals.extend(targets.view(-1).cpu().numpy())
            times.extend(time_data)

        times = np.exp(times)  # Convert log(age) back to age
        df_results = pd.DataFrame({
            "Time": times,
            "Actual": actuals,
            "Predicted": predictions
        })
        df_results.to_csv(results_files[key], index=False)

# =============================================================================
# Hyperparameter Grid Search & Model Selection
# =============================================================================
best_loss = np.inf
best_hyperparams = None
best_model = None

# Grid Search over all combinations
for hyperparams in itertools.product(*[hyperparameters_space[key] for key in hyperparameters_space]):
    hyperparams_dict = dict(zip(hyperparameters_space.keys(), hyperparams))
    model, test_loss, train_loader, test_loader = train_and_evaluate_model(hyperparams_dict)
    if test_loss < best_loss:
        best_loss = test_loss
        print(f"New best loss: {best_loss}, Hyperparameters: {hyperparams_dict}")
        best_hyperparams = hyperparams_dict
        best_model_state = model.state_dict()

# Load and save the best model
if best_model_state is not None:
    best_model = CnnRegressor(Data_x.shape[1], 1,
                              best_hyperparams['conv1_out_channels'],
                              best_hyperparams['conv2_out_channels'],
                              best_hyperparams['linear1_out_features'],
                              best_hyperparams['linear2_out_features']).to(device)
    best_model.load_state_dict(best_model_state)
    print(f"Best Hyperparameters: {best_hyperparams}, Loss: {best_loss}")
    save_model_and_results(best_model, train_loader, test_loader, best_hyperparams)
