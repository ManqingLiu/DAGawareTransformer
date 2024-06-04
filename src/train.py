from argparse import ArgumentParser
import json
import time
from typing import Dict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from src.dataset import CausalDataset, PredictionTransformer
from src.models.transformer_model import DAGTransformer, causal_loss_fun
from src.predict import predict
from utils import IPTW_unstabilized, rmse, replace_column_values


def train(model: nn.Module,
          val_data,
          test_data,
          dag,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          test_dataloader: DataLoader,
          config: Dict,
          mask: bool):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = model.to(device)

    train_config = config['training']

    opt = torch.optim.AdamW(model.parameters(),
                            weight_decay=train_config['weight_decay'],
                            lr=train_config['learning_rate'])

    num_epochs = train_config['num_epochs']
    wandb.init(project="DAG transformer",
               entity="mliu7",
               config=config)
    for epoch in tqdm(range(num_epochs)):
        for _, batch in train_dataloader:
            opt.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch, mask=mask)
            batch_loss, batch_items = causal_loss_fun(outputs, batch, weight=None, return_items=True)
            for item in batch_items.keys():
                wandb.log({f"train_{item}": batch_items[item]})

            batch_loss.backward()
            opt.step()
            wandb.log({'training loss': batch_loss.item()})


        model.eval()
        predictions = []
        with torch.no_grad():
            val_loss = 0
            for _, batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch, mask=mask)
                batch_loss, batch_items = causal_loss_fun(outputs, batch, weight=None)
                for item in batch_items.keys():
                    wandb.log({f"val_{item}": batch_items[item]})
                val_loss += batch_loss.item()
            avg_val_loss = val_loss / len(batch)
            wandb.log({'validation loss': avg_val_loss})


            for _, batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch, mask=mask)
                batch_predictions = []
                for output_name in outputs.keys():
                    # Detach the outputs and move them to cpu
                    output = outputs[output_name].cpu().numpy()
                    output = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
                    # Append the reshaped output to batch_predictions
                    batch_predictions.append(output)
                # concatenate the batch predictions along the second axis
                batch_predictions = np.concatenate(batch_predictions, axis=1)
                predictions.append(batch_predictions)

        # assign column names to the predictions_df
        dataset = CausalDataset(val_data, dag)
        predictions = np.concatenate(predictions, axis=0)
        prediction_transformer = PredictionTransformer(dataset.bin_edges)
        transformed_predictions = prediction_transformer.transform(predictions)
        predictions_final = pd.concat([val_data, transformed_predictions], axis=1)
        ATE_true = val_data['y1'].mean() - val_data['y0'].mean()
        ATE_IPTW = IPTW_unstabilized(predictions_final['t'], predictions_final['y'], predictions_final['t_prob'])
        rmse_IPTW = rmse(ATE_IPTW, ATE_true)
        wandb.log({'RMSE from unstabilized IPTW (val)': rmse_IPTW})
        wandb.log({'average predicted t (val)': predictions_final['t_prob'].mean()})

        model.eval()
        predictions = []
        with torch.no_grad():
            for _, batch in test_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                test_outputs = model(batch, mask=mask)
                batch_predictions = []
                for output_name in outputs.keys():
                    # Detach the outputs and move them to cpu
                    output = test_outputs[output_name].cpu().numpy()
                    output = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
                    # Append the reshaped output to batch_predictions
                    batch_predictions.append(output)
                # concatenate the batch predictions along the second axis
                batch_predictions = np.concatenate(batch_predictions, axis=1)
                predictions.append(batch_predictions)

        # assign column names to the predictions_df
        dataset = CausalDataset(test_data, dag)
        predictions = np.concatenate(predictions, axis=0)
        prediction_transformer = PredictionTransformer(dataset.bin_edges)
        transformed_predictions = prediction_transformer.transform(predictions)
        predictions_final = pd.concat([test_data, transformed_predictions], axis=1)
        ATE_true = test_data['y1'].mean() - test_data['y0'].mean()
        ATE_IPTW = IPTW_unstabilized(predictions_final['t'], predictions_final['y'], predictions_final['t_prob'])
        rmse_IPTW = rmse(ATE_IPTW, ATE_true)
        wandb.log({'RMSE from unstabilized IPTW (test)': rmse_IPTW})
        wandb.log({'average predicted t (test)': predictions_final['t_prob'].mean()})

        model.train()

    wandb.finish()

    return model




