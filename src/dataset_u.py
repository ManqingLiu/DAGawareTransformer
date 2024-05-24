from torch.utils.data import Dataset
import torch
import numpy as np

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from argparse import ArgumentParser
from utils import *
from sklearn.model_selection import train_test_split
from numpy.random import rand, seed, multivariate_normal
from scipy.stats import norm
from scipy.linalg import cholesky

import json

class CausalDataset(Dataset):
    def __init__(self, data, dag, n_uniform_variables=0, add_u=False, seed_value=0):
        self.data = data
        self.dag = dag
        self.bin_edges = {}

        self.num_nodes = len(self.dag['nodes'])
        self.dag['node_ids'] = dict(zip(self.dag['nodes'], range(self.num_nodes)))

        # Add uniform variables before binning columns
        if n_uniform_variables > 0:
            self.add_uniform_variables(n_uniform_variables, add_u, seed_value)

        self.bin_columns()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx].copy()
        return sample

    def add_uniform_variables(self, n, add_u, seed_value):
        """
        Add n variables u_1 to u_n following a uniform distribution to the original data.
        """
        if add_u:
            # Generate a positive definite correlation matrix
            corr_matrix = np.eye(n)  # start with an identity matrix
            corr_matrix[corr_matrix == 0] = 0.5  # off-diagonal correlations are 0.5

            # Cholesky decomposition
            #lower_triangular = cholesky(corr_matrix)

            # Generate independent normal random variables
            np.random.seed(seed_value)  # for reproducibility
            normal_vars = multivariate_normal(np.zeros(n), corr_matrix, size=len(self.data))

            # Transform the normal variables to uniform variables using the CDF
            #uniform_vars = norm.cdf(normal_vars)

            uniform_vars = np.random.uniform(0, 1, (len(self.data), n))

            # pass uniform_vars to logits to map to a real line
            uniform_vars = np.log(uniform_vars / (1 - uniform_vars))


            # Add the uniform variables to the data
            for i in range(n):
                self.data[f'U{i + 1}'] = uniform_vars[:, i]

            # Update the dag to include the new variables
            for i in range(1, n + 1):  # start from 1
                self.dag['nodes'].append(f'U{i}')
                self.dag['node_ids'][f'U{i}'] = len(self.dag['node_ids'])

        # Re-bin the columns to include the new variables
        # self.bin_columns()

    def collate_fn(self, batch_list):
        batch = {}
        for node in self.dag['nodes']:
            if node in batch_list[0]:  # Check if the node exists in the DataFrame
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
    def __init__(self, dag):
        #self.bin_edges = bin_edges
        #self.bin_midpoints = {k: (v[1:] + v[:-1]) / 2 for k, v in bin_edges.items()}
        self.dag = dag

    def transform(self, predictions):
        transformed_predictions = pd.DataFrame()

        # Iterate over all columns
        for i, column in enumerate(self.dag['output_nodes']):
            column_predictions = predictions[:, i * 2:(i + 1) * 2]  # Assuming each column has two prediction values
            column_prob = column_predictions[:, 1]  # Probability of column=1
            transformed_predictions[column + '_prob'] = column_prob

            # Add column_hat which is the predicted binary value based on the predicted probabilities
            column_hat = (column_prob > 0.5).astype(int)
            transformed_predictions[column] = column_hat

        return transformed_predictions

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dag', type=str, required=True)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--train_output_file', type=str, required=True)
    parser.add_argument('--holdout_output_file', type=str, required=True)

    args = parser.parse_args()
    data = pd.read_csv(args.data_file)
    print(data.columns)

    with open(args.dag) as f:
        print(f'Loading dag file from {args.dag}')
        dag = json.load(f)

    #np.random.seed(253)
    #dataset = CausalDataset(data, dag, n_uniform_variables=5)
    #data = data[dag['nodes']]
    #print(dataset.data.columns)
    # split to train and holdout set
    train_data, holdout_data = train_test_split(data, test_size=0.5, random_state=42)
    train_data.to_csv(args.train_output_file, index=False)
    holdout_data.to_csv(args.holdout_output_file, index=False)

