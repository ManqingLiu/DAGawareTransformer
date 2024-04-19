import unittest

import pandas as pd
import numpy as np
import json

import torch
from torch.utils.data import DataLoader

from src.model import DAGTransformer, causal_loss_fun
from src.dataset import CausalDataset

from tqdm import tqdm

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
        self.config_file = '../config/dag/lalonde_psid_dag.json'
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
        model = model.to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)

        outputs = model(batch)

        loss = causal_loss_fun(outputs, batch)
        loss.backward()
        initial_loss = loss.item()
        opt.step()

        outputs = model(batch)
        final_loss = causal_loss_fun(outputs, batch).item()

        self.assertTrue(final_loss < initial_loss)

    def test_dataloader_onestep(self):

        dataset = CausalDataset(self.data_file, self.config_file)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=dataset.collate_fn)

        model = DAGTransformer(self.dag)
        model = model.to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)

        batch = next(iter(dataloader))
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

    def test_dataloader_twoepochs(self):
        batch_size = 64
        data = pd.read_csv(self.data_file)
        data = data[self.dag['nodes']]
        dataset = CausalDataset(data, self.dag)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=4,
                                shuffle=True,
                                collate_fn=dataset.collate_fn)

        model = DAGTransformer(self.dag)
        model = model.to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        for batch in tqdm(dataloader):
            outputs = model(batch)

            batch_loss = []
            for output_name in outputs.keys():
                output = outputs[output_name]
                labels = batch[output_name].squeeze()
                batch_loss.append(loss_fn(output, labels))

            batch_loss  = sum(batch_loss) / len(batch_loss)
            batch_loss.backward()
            opt.step()
            opt.zero_grad()

        # Calculate loss on the entire dataset
        initial_loss = 0
        for batch in dataloader:
            outputs = model(batch)
            batch_loss = 0
            for output_name in outputs.keys():
                output = outputs[output_name]
                labels = batch[output_name].squeeze()
                batch_loss += loss_fn(output, labels)

            batch_loss /= len(outputs.keys())
            initial_loss += batch_loss

        initial_loss /= len(dataloader)

        for batch in tqdm(dataloader):
            outputs = model(batch)

            batch_loss = causal_loss_fun(outputs, batch)
            batch_loss.backward()
            opt.step()
            opt.zero_grad()

        final_loss = 0
        for batch in dataloader:
            outputs = model(batch)
            batch_loss = 0
            for output_name in outputs.keys():
                output = outputs[output_name]
                labels = batch[output_name].squeeze()
                batch_loss += loss_fn(output, labels)

            batch_loss /= len(outputs.keys())
            final_loss += batch_loss

        final_loss /= len(dataloader)
        print(f'Initial loss: {initial_loss.item()}')
        print(f'Final loss: {final_loss.item()}')

        self.assertTrue(final_loss.item() < initial_loss.item())

if __name__ == '__main__':
    unittest.main()
