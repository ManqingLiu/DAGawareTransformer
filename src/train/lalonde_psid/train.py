from typing import Dict
import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from src.models.transformer_model import causal_loss_fun, pdist2sq, safe_sqrt, wasserstein, CFR_loss
from src.train.lalonde_psid.train_metrics import calculate_metrics, create_metric_plots


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    train_config: Dict,
    imbalance_loss_weight: float
) -> nn.Module:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        weight_decay=train_config["weight_decay"],
        lr=train_config["learning_rate"],
    )

    bin_left_edges = {k: torch.tensor(v[:-1], dtype=torch.float32).to(device) for k, v in train_dataloader.dataset.bin_edges.items()}

    for epoch in range(train_config["num_epochs"]):
        for batch_raw, batch_binned in train_dataloader:
            opt.zero_grad()
            batch = {k: v.to(device) for k, v in batch_binned.items()}
            outputs = model(batch, mask=train_config["dag_attention_mask"])

            transformed_outputs = {}
            for output_name, output in outputs.items():
                softmax_output = torch.softmax(output, dim=1)
                if output_name == 'y':
                    pred_y = torch.sum(softmax_output * bin_left_edges[output_name].to(device), dim=1, keepdim=True)
                    transformed_outputs[output_name] = pred_y
                else:
                    transformed_outputs[output_name] = softmax_output[:, 1]
                    h_rep = output


            t = batch_raw['t']
            y = batch_raw['y']
            e = transformed_outputs['t']
            y_ = torch.squeeze(transformed_outputs['y'])
            h_rep_norm = h_rep / safe_sqrt(torch.sum(h_rep ** 2, dim=1, keepdim=True))

            # Use CFR loss function
            batch_loss, batch_items = CFR_loss(t, e, y, y_, h_rep_norm, imbalance_loss_weight, return_items=True)

            batch_loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            wandb.log({f"Train: counterfactual loss": batch_loss.item()})


        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch_raw, batch_binned in val_dataloader:
                batch = {k: v.to(device) for k, v in batch_binned.items()}
                outputs = model(batch, mask=train_config["dag_attention_mask"])

                transformed_outputs = {}
                for output_name, output in outputs.items():
                    softmax_output = torch.softmax(output, dim=1)
                    if output_name == 'y':
                        pred_y = torch.sum(softmax_output * bin_left_edges[output_name].to(device), dim=1, keepdim=True)
                        transformed_outputs[output_name] = pred_y
                    else:
                        transformed_outputs[output_name] = softmax_output[:, 1]
                        h_rep = output

                t = batch_raw['t']
                y = batch_raw['y']
                e = transformed_outputs['t']
                y_ = torch.squeeze(transformed_outputs['y'])
                h_rep_norm = h_rep / safe_sqrt(torch.sum(h_rep ** 2, dim=1, keepdim=True))

                # Use CFR loss function
                val_batch_loss, val_batch_items = CFR_loss(t, e, y, y_, h_rep_norm, imbalance_loss_weight,
                                                           return_items=True)

                val_loss += val_batch_loss.item()
            wandb.log({f"Val: counterfactual loss": val_loss/len(batch)})

        model.train()

    return model
