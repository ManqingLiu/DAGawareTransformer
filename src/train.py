from argparse import ArgumentParser
import json
import pandas as pd

import torch.nn as nn
from torch.utils.data import DataLoader
import torch

from src.models.transformer_model import DAGTransformer, causal_loss_fun
from src.dataset import CausalDataset

from typing import Dict

from tqdm import tqdm
import time

from sklearn.model_selection import train_test_split

import wandb
from matplotlib import pyplot as plt


def train(model: nn.Module,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          config: Dict,
          mask: bool,
          save_model: bool,
          model_file: str):

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
    for epoch in tqdm(range(num_epochs)):
        for _, batch in train_dataloader:
            opt.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch, mask=mask)
            #for output_name in outputs.keys():
            #    print(f"Shape of {output_name}: {outputs[output_name].shape}")
            batch_loss, batch_items = causal_loss_fun(outputs, batch, return_items=True)
            for item in batch_items.keys():
                wandb.log({item: batch_items[item]})

            batch_loss.backward()
            opt.step()
            wandb.log({'training loss': batch_loss.item()})

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for _, batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch, mask=mask)
                batch_loss, _ = causal_loss_fun(outputs, batch)
                for item in batch_items.keys():
                    wandb.log({item: batch_items[item]})
                val_loss += batch_loss.item()
            avg_val_loss = val_loss / len(val_dataloader)
            wandb.log({'validation loss': avg_val_loss})

    wandb.finish()
    #print(f'Saving model to {model_file}')
    if save_model==True:
        torch.save(model.state_dict(), model_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dag', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mask', type=bool, default=True)
    parser.add_argument('--data_train_file', type=str, required=True)
    parser.add_argument('--data_holdout_file', type=str, required=True)
    parser.add_argument('--model_train_file', type=str, required=True)
    parser.add_argument('--model_holdout_file', type=str, required=True)

    args = parser.parse_args()

    with open(args.dag) as f:
        print(f'Loading dag file from {args.dag}')
        dag = json.load(f)

    with open(args.config) as f:
        print(f'Loading config file from {args.config}')
        config = json.load(f)

    train_config = config['training']
    model_config = config['model']

    # Move all of this to a utils function for load_dag and load_data
    num_nodes = len(dag['nodes'])
    dag['node_ids'] = dict(zip(dag['nodes'], range(num_nodes)))
    print(dag)

    model = DAGTransformer(dag=dag,
                           **model_config)

    train_data = pd.read_csv(args.data_train_file)
    train_data = train_data[dag['nodes']]
    train_dataset = CausalDataset(train_data, dag)

    batch_size = train_config['batch_size']
    train_dataloader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=train_dataset.collate_fn)

    holdout_data = pd.read_csv(args.data_holdout_file)
    holdout_data = holdout_data[dag['nodes']]
    holdout_dataset = CausalDataset(holdout_data, dag)
    holdout_dataloader = DataLoader(holdout_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=holdout_dataset.collate_fn)
    # Before training for train file
    start_time = time.time()
    train(model,
          train_dataloader,
          holdout_dataloader,
          train_config,
          mask=args.mask,
          save_model=True,
          model_file=args.model_train_file)
    print('Done training.')

    data = pd.read_csv(args.data_holdout_file)
    data = data[dag['nodes']]
    dataset = CausalDataset(data, dag)

    batch_size = train_config['batch_size']
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=dataset.collate_fn)

    train(model,
          holdout_dataloader,
          train_dataloader,
          train_config,
          mask=args.mask,
          save_model=True,
          model_file=args.model_holdout_file)
    print('Done training.')
    # After training for holdout file
    end_time = time.time()

    # Calculate and print the total wall time
    total_wall_time = end_time - start_time
    # Convert the total wall time to minutes and seconds
    minutes, seconds = divmod(total_wall_time, 60)
    print(f"Total wall time used: {minutes} minutes and {seconds} seconds")

