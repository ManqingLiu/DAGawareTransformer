from torch.utils.data import Dataset
import torch
import numpy as np

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from argparse import ArgumentParser
from utils import *
from sklearn.model_selection import train_test_split

import json

class CausalDataset(Dataset):
    def __init__(self, data, dag):
        self.data = data
        self.dag = dag
        self.bin_edges = {}

        self.num_nodes = len(self.dag['nodes'])
        self.dag['node_ids'] = dict(zip(self.dag['nodes'], range(self.num_nodes)))

        self.bin_columns()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

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
                self.data[column] = binner.fit_transform(self.data[column].values.reshape(-1, 1)).flatten()
                self.data[column] = self.data[column].astype(int)
                self.bin_edges[column] = binner.bin_edges_[0]
            elif num_bins == 2:
                self.data[column] = pd.cut(self.data[column], bins=2, labels=False)


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
    #train_data_predict_A1 = replace_column_values(train_data, 't', 1)
    #holdout_data_predict_A1 = replace_column_values(holdout_data, 't', 1)
    #train_data_predict_A0 = replace_column_values(train_data, 't', 0)
    #holdout_data_predict_A0 = replace_column_values(holdout_data, 't', 0)

    train_data.to_csv(args.train_output_file, index=False)
    holdout_data.to_csv(args.holdout_output_file, index=False)
    #train_data_predict_A1.to_csv(args.train_A1_output_file, index=False)
    #holdout_data_predict_A1.to_csv(args.holdout_A1_output_file, index=False)
    #train_data_predict_A0.to_csv(args.train_A0_output_file, index=False)
    #holdout_data_predict_A0.to_csv(args.holdout_A0_output_file, index=False)

