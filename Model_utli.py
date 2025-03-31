# =============================================================================
# Guangshuai Han (Jerry) @ Purdue University, Luna Group
# Utility Module for EMI-Based AI Strength Prediction from Piezoelectric Sensing
# =============================================================================

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from ignite.contrib.metrics.regression.r2_score import R2Score
import torch.nn.functional as F

# =============================================================================
# Class: EMI_loader
# Description: Load single h5 file for EMI signal-based strength prediction
# =============================================================================
class EMI_loader:
    def __init__(self, file_name):
        f = h5py.File(file_name, 'r')
        print(f.keys())
        self.original_freq = np.arange(10, 505, 5)
        self.real_spec = np.array(f['con_sensor_R'][...])
        self.real_B = np.array(f['con_sensor_RB'][...])
        self.supp = np.array(f['supp'][...])
        self.Temp = self.supp[:, 0]
        self.age = self.supp[:, 1]
        self.RMSD = self.supp[:, 2]
        self.log_age = np.log(self.age)
        try:
            self.Label = np.array(f['Label'][...])
        except:
            print('testing file only, label not available')
            pass

    def fea_spec_logage(self):
        spec_CB = []
        for i in range(self.real_B.shape[0]):
            RC_spec = np.hstack([self.real_spec[i], self.log_age[i]])
            RB_spec = np.hstack([self.real_B[i], self.log_age[i]])
            temp_mat = np.vstack([RC_spec, RB_spec]).T
            spec_CB.append(temp_mat)
        return np.array(spec_CB)

    def fea_spec_temp_logage(self):
        spec_CB = []
        for i in range(self.real_B.shape[0]):
            RC_spec = np.hstack([self.real_spec[i], self.Temp[i], self.log_age[i]])
            RB_spec = np.hstack([self.real_B[i], self.Temp[i], self.log_age[i]])
            temp_mat = np.vstack([RC_spec, RB_spec]).T
            spec_CB.append(temp_mat)
        return np.array(spec_CB)

# =============================================================================
# Class: RawLoader
# Description: Load and merge multiple slab .h5 files with frequency slicing
# =============================================================================
class RawLoader:
    def __init__(self, file_list, start_freq=10, end_freq=505):
        self.file_list = file_list
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.real_spec = []
        self.real_B = []
        self.Temp = []
        self.age = []
        self.RMSD = []
        self.Label = []
        self.original_freq = np.arange(10, 505, 5)
        print(self.original_freq.shape)
        self.load_and_merge()
        self.slice_by_freq_range()

    def load_and_merge(self):
        for file_path in self.file_list:
            with h5py.File(file_path, 'r') as f:
                self.real_spec.append(np.array(f['con_sensor_R'][...]))
                self.real_B.append(np.array(f['con_sensor_RB'][...]))
                supp = np.array(f['supp'][...])
                self.Temp.append(supp[:, 0])
                self.age.append(np.log(supp[:, 1]))
                self.RMSD.append(supp[:, 2])
                self.Label.append(np.array(f['Label'][...]))
        self.real_spec = np.concatenate(self.real_spec, axis=0)
        self.real_B = np.concatenate(self.real_B, axis=0)
        self.Temp = np.concatenate(self.Temp, axis=0)
        self.age = np.concatenate(self.age, axis=0)
        self.RMSD = np.concatenate(self.RMSD, axis=0)
        self.Label = np.concatenate(self.Label, axis=0)

    def slice_by_freq_range(self):
        start_index = np.searchsorted(self.original_freq, self.start_freq, side='left')
        end_index = np.searchsorted(self.original_freq, self.end_freq, side='right')
        self.real_spec = self.real_spec[:, start_index:end_index]
        self.real_B = self.real_B[:, start_index:end_index]
        self.original_freq = self.original_freq[start_index:end_index]

    def fea_spec_logage(self):
        spec_CB = []
        for i in range(self.real_B.shape[0]):
            RC_spec = np.hstack([self.real_spec[i], self.age[i]])
            RB_spec = np.hstack([self.real_B[i], self.age[i]])
            temp_mat = np.vstack([RC_spec, RB_spec]).T
            spec_CB.append(temp_mat)
        return np.array(spec_CB)

    def fea_spec_temp_logage(self):
        spec_CB = []
        for i in range(self.real_B.shape[0]):
            RC_spec = np.hstack([self.real_spec[i], self.Temp[i], self.age[i]])
            RB_spec = np.hstack([self.real_B[i], self.Temp[i], self.age[i]])
            temp_mat = np.vstack([RC_spec, RB_spec]).T
            spec_CB.append(temp_mat)
        return np.array(spec_CB)

    def fea_spec_temp(self):
        spec_CB = []
        for i in range(self.real_B.shape[0]):
            RC_spec = np.hstack([self.real_spec[i], self.Temp[i]])
            RB_spec = np.hstack([self.real_B[i], self.Temp[i]])
            temp_mat = np.vstack([RC_spec, RB_spec]).T
            spec_CB.append(temp_mat)
        return np.array(spec_CB)

    def fea_spec_temp_logage_NB(self):
        spec_CB = []
        for i in range(self.real_B.shape[0]):
            RC_spec = np.hstack([self.real_spec[i], self.Temp[i], self.age[i]])
            RB_spec = np.hstack([self.real_spec[i], self.Temp[i], self.age[i]])
            temp_mat = np.vstack([RC_spec, RB_spec]).T
            spec_CB.append(temp_mat)
        return np.array(spec_CB)

    def _freq_slice(self):
        start_index = np.searchsorted(self.original_freq, self.start_freq, side='left')
        end_index = np.searchsorted(self.original_freq, self.end_freq, side='right')
        return slice(start_index, end_index)

    def fea_spec_temp_logage_with_meta(self):
        spec_CB = []
        source_files = []
        source_indices = []
        rc_max_values = []
        temp_values = []

        for file_path in self.file_list:
            with h5py.File(file_path, 'r') as f:
                num_samples = f['con_sensor_R'].shape[0]
                file_name = file_path.split('/')[-1]
                for i in range(num_samples):
                    RC_spec = np.hstack([f['con_sensor_R'][i][self._freq_slice()], f['supp'][i, 0], np.log(f['supp'][i, 1])])
                    RB_spec = np.hstack([f['con_sensor_RB'][i][self._freq_slice()], f['supp'][i, 0], np.log(f['supp'][i, 1])])
                    temp_mat = np.vstack([RC_spec, RB_spec]).T
                    spec_CB.append(temp_mat)
                    source_files.append(file_name)
                    source_indices.append(i)
                    rc_max_values.append(np.max(f['con_sensor_R'][i][self._freq_slice()]))
                    temp_values.append(f['supp'][i, 0])

        return (np.array(spec_CB), self.Label, 
                np.array(source_files), np.array(source_indices), 
                np.array(rc_max_values), np.array(temp_values))

# =============================================================================
# Function: model_loss
# Description: Evaluate model loss and R² score over dataset (optionally train)
# =============================================================================
def model_loss(model, dataset, train=False, optimizer=None):
    performance = nn.MSELoss()
    score_metric = R2Score()
    avg_loss = 0
    avg_score = 0
    count = 0
    for input, output in iter(dataset):
        predictions = model(input)
        loss = performance(predictions, output)
        score_metric.update([predictions, output])
        score = score_metric.compute()
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss += loss.item()
        avg_score += score
        count += 1
    return avg_loss / count, avg_score / count

# =============================================================================
# Model: CnnRegressor
# Description: Convolutional neural network for EMI feature regression
# =============================================================================
class CnnRegressor(nn.Module):
    def __init__(self, inputs, outputs, conv1_out_channels=256, conv2_out_channels=128,
                 linear1_out_features=64, linear2_out_features=32, BN_id=False):
        super(CnnRegressor, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.BN_id = BN_id

        self.input_layer = nn.Conv1d(inputs, conv1_out_channels, 2)
        self.max_pooling_layer = nn.MaxPool1d(1)
        self.conv_layer1 = nn.Conv1d(conv1_out_channels, conv2_out_channels, 1)
        self.conv_layer2 = nn.Conv1d(conv2_out_channels, 128, 1)
        self.bn_layer = nn.BatchNorm1d(num_features=128)

        self.flatten_layer = nn.Flatten()
        self.linear_layer_1 = nn.Linear(128, linear1_out_features)
        self.linear_layer_2 = nn.Linear(linear1_out_features, linear2_out_features)
        self.linear_layer_3 = nn.Linear(linear2_out_features + 2, 16)  # Add extra features
        self.output_layer = nn.Linear(16, outputs)

    def forward(self, input):
        temp_BS = input.shape[0]
        supp = input[:, -2:, 1]
        input = input.reshape((temp_BS, self.inputs, -1))
        output = F.relu(self.input_layer(input))
        output = self.max_pooling_layer(output)
        output = F.relu(self.conv_layer1(output))
        output = F.relu(self.conv_layer2(output))
        if self.BN_id:
            output = output.view(temp_BS, 128, 1, 1)
            output = self.bn_layer(output)
            output = output.squeeze(dim=3)
        output = self.flatten_layer(output)
        output = F.relu(self.linear_layer_1(output))
        output = F.relu(self.linear_layer_2(output))
        output = torch.cat((output, supp), dim=1)
        output = F.relu(self.linear_layer_3(output))
        output = self.output_layer(output)
        return output

# =============================================================================
# Model: SimpleNNRegressor
# Description: Simple feedforward neural network for comparison
# =============================================================================
class SimpleNNRegressor(nn.Module):
    def __init__(self, inputs, outputs, linear1_out_features=64, linear2_out_features=32):
        super(SimpleNNRegressor, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.linear_layer_1 = nn.Linear(inputs * 2, linear1_out_features)
        self.linear_layer_2 = nn.Linear(linear1_out_features, linear2_out_features)
        self.output_layer = nn.Linear(linear2_out_features, outputs)

    def forward(self, input):
        input = input.view(input.shape[0], -1)
        output = F.relu(self.linear_layer_1(input))
        output = F.relu(self.linear_layer_2(output))
        output = self.output_layer(output)
        return output
