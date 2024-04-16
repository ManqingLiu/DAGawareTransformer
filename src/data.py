from torch.utils.data import Dataset
import torch

import pandas as pd
import json

class CausalDataset(Dataset):
    def __init__(self, data, dag):
        self.data = data
        self.dag = dag

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
            if num_bins > 2:
                self.data[column] = pd.qcut(self.data[column], q=num_bins, duplicates='drop', labels=False)
            elif num_bins == 2:
                self.data[column] = pd.cut(self.data[column], bins=2, labels=False)
