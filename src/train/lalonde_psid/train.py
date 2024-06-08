from typing import Dict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from src.models.transformer_model import causal_loss_fun
from src.train.lalonde_psid.train_metrics import calculate_metrics, create_metric_plots


def train(
    model: nn.Module,
    val_data: pd.DataFrame,
    train_data: pd.DataFrame,
    dag: Dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    train_config: Dict,
    random_seed: int,
) -> nn.Module:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        weight_decay=train_config["weight_decay"],
        lr=train_config["learning_rate"],
    )

    # for epoch in tqdm(range(train_config["num_epochs"])):
    for epoch in range(train_config["num_epochs"]):
        predictions = []
        for _, batch in train_dataloader:
            opt.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch, mask=train_config["dag_attention_mask"])

            batch_predictions = []
            for output_name in outputs.keys():
                # Detach the outputs and move them to cpu
                output = outputs[output_name].detach().numpy()
                output = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
                # Append the reshaped output to batch_predictions
                batch_predictions.append(output)
            # concatenate the batch predictions along the second axis
            batch_predictions = np.concatenate(batch_predictions, axis=1)
            predictions.append(batch_predictions)

            batch_loss, batch_items = causal_loss_fun(outputs, batch)
            for item in batch_items.keys():
                wandb.log({f"Train: {item} loss": batch_items[item]})

            batch_loss.backward()
            opt.step()
            wandb.log({f"Train: average loss": batch_loss.item()})

        wandb.log(
            calculate_metrics(
                train_data,
                dag,
                np.concatenate(predictions, axis=0),
                prefix="Train",
                random_seed=random_seed,
            )
        )
        plot_dict = create_metric_plots(
            train_data,
            dag,
            np.concatenate(predictions, axis=0),
            prefix="Train",
            suffix=epoch,
            random_seed=random_seed,
        )
        wandb.log(
            {
                plot_name: wandb.Image(image_path)
                for plot_name, image_path in plot_dict.items()
            }
        )

        model.eval()
        predictions = []
        with torch.no_grad():
            val_loss = 0
            for _, batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch, mask=train_config["dag_attention_mask"])

                batch_predictions = []
                for output_name in outputs.keys():
                    # Detach the outputs and move them to cpu
                    output = outputs[output_name].cpu().numpy()
                    output = np.exp(output) / np.sum(
                        np.exp(output), axis=1, keepdims=True
                    )
                    # Append the reshaped output to batch_predictions
                    batch_predictions.append(output)
                # concatenate the batch predictions along the second axis
                batch_predictions = np.concatenate(batch_predictions, axis=1)
                predictions.append(batch_predictions)

                batch_loss, batch_items = causal_loss_fun(outputs, batch)
                for item in batch_items.keys():
                    wandb.log({f"Val: {item} loss": batch_items[item]})
                val_loss += batch_loss.item()

            wandb.log({f"Val: average loss": val_loss / len(batch)})

        wandb.log(
            calculate_metrics(
                val_data,
                dag,
                np.concatenate(predictions, axis=0),
                prefix="Val",
                random_seed=random_seed,
            )
        )
        plot_dict = create_metric_plots(
            val_data,
            dag,
            np.concatenate(predictions, axis=0),
            prefix="Val",
            suffix=epoch,
            random_seed=random_seed,
        )
        wandb.log(
            {
                plot_name: wandb.Image(image_path)
                for plot_name, image_path in plot_dict.items()
            }
        )
        model.train()

    return model
