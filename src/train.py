from argparse import ArgumentParser
import json
import pandas as pd

import torch.nn as nn
from torch.utils.data import DataLoader
import torch

from model import DAGTransformer, causal_loss_fun
from dataset import CausalDataset

from typing import Dict

import wandb

def train(model: nn.Module, dataloader: DataLoader, config: Dict):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = model.to(device)

    wandb.init(project="DAG transformer",
               config=config,
               tags=[config['project_tag']])

    opt = torch.optim.AdamW(model.parameters(),
                            weight_decay=config['weight_decay'],
                            lr=config['learning_rate'])
    num_epochs = config['num_epochs']
    for epoch in range(num_epochs):
        for batch in dataloader:
            opt.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            batch_loss, batch_items = causal_loss_fun(outputs, batch, return_items=True)
            for item in batch_items.keys():
                wandb.log({item: batch_items[item]})

            batch_loss.backward()
            opt.step()
            wandb.log({'loss': batch_loss})



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dag', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_file', type=str, required=True)

    args = parser.parse_args()

    with open(args.dag) as f:
        print(f'Loading dag file from {args.dag}')
        dag = json.load(f)

    with open(args.config) as f:
        print(f'Loading config file from {args.config}')
        config = json.load(f)

    # Move all of this to a utils function for load_dag and load_data
    num_nodes = len(dag['nodes'])
    dag['node_ids'] = dict(zip(dag['nodes'], range(num_nodes)))

    model = DAGTransformer(dag)

    data = pd.read_csv(args.data_file)
    data = data[dag['nodes']]
    dataset = CausalDataset(data, dag)

    batch_size = config['batch_size']
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=dataset.collate_fn)

    train(model, dataloader, config)
    print('Done training.')

