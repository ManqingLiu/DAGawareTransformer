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
        y_expected_value = y_predictions[:, 1]

        #y_expected_value = np.sum(y_predictions * self.bin_midpoints['y'], axis=1)

        transformed_predictions = pd.DataFrame({
            't_prob': t_prob,
            'pred_y': y_expected_value
        })

        return transformed_predictions

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dag', type=str, required=True)
    parser.add_argument('--data_file', type=str, required=True)

    args = parser.parse_args()
    data = pd.read_csv(args.data_file)

    with open(args.dag) as f:
        print(f'Loading dag file from {args.dag}')
        dag = json.load(f)
    '''
    # Define the new names
    new_names = ["eclamp", "gestatcat1", "gestatcat2", "gestatcat3", "gestatcat4",
                 "gestatcat5", "gestatcat6", "gestatcat7", "gestatcat8", "gestatcat9",
                 "gestatcat10", "gestatcat1.1", "gestatcat2.1", "gestatcat3.1", "bord",
                 "gestatcat4.1", "gestatcat6.1", "gestatcat7.1", "gestatcat8.1",
                 "gestatcat9.1", "gestatcat10.1", "gestatcat1.2", "gestatcat2.2",
                 "gestatcat3.2", "gestatcat4.2", "gestatcat5.1", "gestatcat6.2",
                 "gestatcat7.2", "gestatcat8.2", "gestatcat5.2", "gestatcat9.2",
                 "gestatcat10.2", "othermr", "dmar", "csex", "cardiac", "uterine",
                 "lung", "diabetes", "herpes", "anemia", "hydra", "chyper", "phyper",
                 "incervix", "pre4000", "preterm", "renal", "rh", "hemo", "tobacco",
                 "alcohol", "orfath", "adequacy", "drink5", "mpre5", "meduc6", "mrace",
                 "ormoth", "frace", "birattnd", "stoccfipb_reg", "mplbir_reg", "cigar6",
                 "mager8", "pldel", "brstate_reg", "feduc6", "dfageq", "nprevistq",
                 "data_year", "crace", "birmon", "dtotord_min", "dlivord_min", "t", "y"]


    # Initialize new_input_nodes with all new names
    new_input_nodes = {new_name: {} for new_name in new_names}

    # Update num_categories for each node
    for new_name, old_node in zip(new_names, dag['input_nodes'].values()):
        new_input_nodes[new_name] = old_node

    dag['input_nodes'] = new_input_nodes

    # Iterate over the input_nodes
    for node in dag['input_nodes']:
        # Detect the variable type
        var_type = detect_variable_type(data, node)

        # Set num_categories based on the variable type
        if var_type == 'continuous':
            dag['input_nodes'][node]['num_categories'] = 10
        elif var_type == 'binary':
            dag['input_nodes'][node]['num_categories'] = 2

    # Get all node names
    all_nodes = list(dag['input_nodes'].keys())

    # Iterate over the input_nodes
    for node in all_nodes:
        # For each node, create a list of all other nodes, excluding 'y'
        if node != 't' and node != 'y':
            dag['edges'][node] = [other_node for other_node in all_nodes]
        # For the node 't', set its list to only include 'y'
        else:
            dag['edges'][node] = ['y']

    # Write the updated dictionary back to the JSON file
    with open('../config/dag/twins_dag.json', 'w') as f:
        json.dump(dag, f, indent=4)
    '''

    data = data[dag['nodes']]
    # split to train and holdout set
    train_data, holdout_data = train_test_split(data, test_size=0.5, random_state=42)
    train_data.to_csv('data/realcause_datasets/twins/sample0/train/twins_train.csv', index=False)
    holdout_data.to_csv('data/realcause_datasets/twins/sample0/holdout/twins_holdout.csv', index=False)

