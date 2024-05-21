from torch.utils.data import Dataset
import torch
import numpy as np
from src.data.ate.data_class import PVTrainDataSet

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from argparse import ArgumentParser
from utils import *
from sklearn.model_selection import train_test_split

import json

class CausalDataset(Dataset):
    def __init__(self, data, dag):
        self.data = data
        
        if isinstance(self.data, pd.DataFrame):
            self.data_binned = self.data.copy()
        elif isinstance(self.data, dict):
            self.data_binned = {k: None for k, _ in self.data.items()}
        
        self.dag = dag
        self.bin_edges = {}

        self.num_nodes = len(self.dag['nodes'])
        self.dag['node_ids'] = dict(zip(self.dag['nodes'], range(self.num_nodes)))

        if isinstance(self.data, pd.DataFrame):
            self.bin_columns()
        elif isinstance(self.data, dict):
            self.bin_columns_for_ndarray()

    def __len__(self):
        if isinstance(self.data, pd.DataFrame):
            return len(self.data)
        elif isinstance(self.data, dict):
            first_key = next(iter(self.data))
            return len(self.data[first_key])

    def __getitem__(self, idx):
        if isinstance(self.data, pd.DataFrame):
            return self.data.iloc[idx], self.data_binned.iloc[idx]
        elif isinstance(self.data, dict):
            return {key: self.data[key][idx] for key in self.data}, 
        {key: self.data_binned[key][idx] for key in self.data_binned}

    def collate_fn(self, batch_list):
        batch = {}
        for node in self.dag['input_nodes']:
            node_data = torch.stack([torch.tensor(row[node]) for row in batch_list])
            batch[node] = node_data.unsqueeze(1)
        return batch

    def bin_columns(self):
        for column, params in self.dag['input_nodes'].items():
            num_bins = params['num_categories']
            binner = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='uniform')
            if num_bins > 2:
                self.data_binned[column] = binner.fit_transform(self.data[column].values.reshape(-1, 1)).flatten()
                self.data_binned[column] = self.data_binned[column].astype(int)
                self.bin_edges[column] = binner.bin_edges_[0]
            elif num_bins == 2:
                self.data_binned[column] = pd.cut(self.data[column], bins=2, labels=False)

    def bin_columns_for_ndarray(self):
        for column, params in self.dag['input_nodes'].items():
            num_bins = params['num_categories']
            binner = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='uniform')
            if num_bins > 2:
                self.data_binned[column] = binner.fit_transform(self.data[column].reshape(-1, 1)).flatten()
                self.data_binned[column] = self.data_binned[column].astype(int)
                self.bin_edges[column] = binner.bin_edges_[0]
            elif num_bins == 2:
                self.data_binned[column] = pd.cut(self.data[column], bins=2, labels=False).to_numpy()

    def to_pvtraindataset(self):
        treatment = self.data['treatment'] if 'treatment' in self.data else None
        treatment_proxy = self.data['treatment_proxy'] if 'treatment_proxy' in self.data else None
        outcome_proxy = self.data['outcome_proxy'] if 'outcome_proxy' in self.data else None
        outcome = self.data['outcome'] if 'outcome' in self.data else None
        backdoor = self.data['backdoor'] if 'backdoor' in self.data else None

        return PVTrainDataSet(
            treatment=treatment,
            treatment_proxy=treatment_proxy,
            outcome_proxy=outcome_proxy,
            outcome=outcome,
            backdoor=backdoor
        )
    
    def get_bin_left_edges(self):
        return {k: v[:-1] for k, v in self.bin_edges.items()}

class PredictionTransformer:
    def __init__(self, bin_edges):
        self.bin_edges = bin_edges
        self.bin_midpoints = {k: (v[1:] + v[:-1]) / 2 for k, v in bin_edges.items()}

    def transform(self, predictions):
        t_predictions = predictions[:, :2]
        y_predictions = predictions[:, 2:]

        t_prob = t_predictions[:, 1]  # Probability of t=1
        # Calculate y_expected_value based on the number of columns in y_predictions
        if y_predictions.shape[1] == 2:
            y_expected_value = y_predictions[:, 1]
        elif y_predictions.shape[1] > 2:
            y_expected_value = np.sum(y_predictions * self.bin_midpoints['y'], axis=1)

        transformed_predictions = pd.DataFrame({
            't_prob': t_prob,
            'pred_y': y_expected_value
        })

        return transformed_predictions

    def transform_proximal(self, predictions, n_sample):
        # To Do: check the shape of predictions
        expected_value = np.sum(predictions * self.bin_midpoints['outcome'], axis=1)

        # Transform the expected_value into a tensor
        transformed_predictions = torch.tensor(expected_value, dtype=torch.float32).view(10, n_sample, 1)

        return transformed_predictions


    def transform_frontdoor(self, predictions):
        # m_predictions is the 3rd and 4th columns of the predictions
        m_predictions = predictions[:, 2:4]
        y_predictions = predictions[:, 4:]

        m_prob = m_predictions[:, 1]  # Probability of m=1
        if y_predictions.shape[1] == 2:
            y_expected_value = y_predictions[:, 1]
        elif y_predictions.shape[1] > 2:
            y_expected_value = np.sum(y_predictions * self.bin_midpoints['Y'], axis=1)

        transformed_predictions = pd.DataFrame({
            'm_prob': m_prob,
            'pred_y': y_expected_value
        })

        return transformed_predictions



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dag', type=str, required=True)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--train_output_file', type=str, required=True)
    parser.add_argument('--holdout_output_file', type=str, required=True)

    args = parser.parse_args()
    data = pd.read_csv(args.data_file)

    with open(args.dag) as f:
        print(f'Loading dag file from {args.dag}')
        dag = json.load(f)

    data = data[dag['nodes']]
    # split to train and holdout set
    train_data, holdout_data = train_test_split(data, test_size=0.5, random_state=42)

    train_data.to_csv(args.train_output_file, index=False)
    holdout_data.to_csv(args.holdout_output_file, index=False)


