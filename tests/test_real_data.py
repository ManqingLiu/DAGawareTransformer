import unittest

import pandas as pd
import numpy as np
import json

import torch

from src.model import DAGTransformer

def bin_columns(df, config):
    for column, params in config.items():
        num_bins = params['num_categories']
        if num_bins > 2:
            df[column] = pd.qcut(df[column], q=num_bins, duplicates='drop', labels=False)
        elif num_bins == 2:
            df[column] = pd.cut(df[column], bins=2, labels=False)
    return df

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.config_file = '../config/lalonde_psid.json'
        self.data_file = '../data/realcause_datasets/lalonde_psid_sample1.csv'
        with open(self.config_file) as f:
            self.dag = json.load(f)
        num_nodes = len(self.dag['nodes'])
        self.dag['node_ids'] = dict(zip(self.dag['nodes'], range(num_nodes)))

        self.data = pd.read_csv(self.data_file)
        self.data = self.data[self.dag['nodes']]

        bin_columns(self.data, self.dag['input_nodes'])

        device_name = 'cpu'
        #if torch.cuda.is_available():
        #    device_name = 'cuda'
        #elif torch.backends.mps.is_available():
        #    device_name = 'mps'
        self.device = torch.device(device_name)

    def test_one_step(self):
        batch_size = 8
        batch = {}
        for node in self.dag['input_nodes']:
            node_data = torch.tensor(self.data[node].values[:batch_size]).unsqueeze(1)
            node_data = node_data.to(self.device)
            batch[node] = node_data

        model = DAGTransformer(self.dag)
        model.to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)

        outputs = model(batch)

        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        total_loss = 0
        for output_name in outputs.keys():
            output = outputs[output_name]
            labels = batch[output_name].squeeze()
            total_loss += loss_fn(output, labels)

        total_loss /= len(outputs.keys())
        initial_loss = total_loss.item()
        total_loss.backward()
        opt.step()

        outputs = model(batch)
        total_loss = 0
        for output_name in outputs.keys():
            output = outputs[output_name]
            labels = batch[output_name].squeeze()
            total_loss += loss_fn(output, labels)

        total_loss /= len(outputs.keys())

        self.assertTrue(total_loss.item() < initial_loss)


if __name__ == '__main__':
    unittest.main()
