from argparse import ArgumentParser
import json
import os
import time
from typing import Dict

from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from src.models.transformer_model import DAGTransformer, causal_loss_fun
from src.predict import predict
from src.train.lalonde_psid.train_metrics import (
    calculate_metrics,
    create_metric_plots,
    images_to_gif,
)


def train(
    model: nn.Module,
    val_data: pd.DataFrame,
    train_data: pd.DataFrame,
    dag: Dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    config: Dict,
    mask: bool,
) -> nn.Module:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    train_config = config["training"]

    opt = torch.optim.AdamW(
        model.parameters(),
        weight_decay=train_config["weight_decay"],
        lr=train_config["learning_rate"],
    )

    run = wandb.init(project="DAG transformer", entity="mliu7", config=config)

    for epoch in tqdm(range(train_config["num_epochs"])):
        predictions = []
        for _, batch in train_dataloader:
            opt.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch, mask=mask)

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
            wandb.log({"Train: average loss": batch_loss.item()})

        wandb.log(
            calculate_metrics(
                train_data, dag, np.concatenate(predictions, axis=0), prefix="Train"
            )
        )
        plot_dict = create_metric_plots(
            train_data,
            dag,
            np.concatenate(predictions, axis=0),
            prefix="Train",
            suffix=epoch,
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
                outputs = model(batch, mask=mask)

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

            wandb.log({"Val: average loss": val_loss / len(batch)})

        wandb.log(
            calculate_metrics(
                val_data, dag, np.concatenate(predictions, axis=0), prefix="Val"
            )
        )
        plot_dict = create_metric_plots(
            val_data,
            dag,
            np.concatenate(predictions, axis=0),
            prefix="Val",
            suffix=epoch,
        )
        wandb.log(
            {
                plot_name: wandb.Image(image_path)
                for plot_name, image_path in plot_dict.items()
            }
        )
        model.train()

    wandb.finish()

    api = wandb.Api()
    run = api.run(run.path)
    for file in run.files():
        if file.name.endswith(".png") and "Train" in file.name:
            file.download(root="train_gif_images", replace=True, exist_ok=True)
        if file.name.endswith(".png") and "Val" in file.name:
            file.download(root="val_gif_images", replace=True, exist_ok=True)

    for split, filepath in {
        "train": "train_gif_images",
        "val": "val_gif_images",
    }.items():
        image_filepaths = []
        image_directory = os.path.join(filepath, "media/images")
        for root, dirs, files in os.walk(image_directory):
            for file in files:
                if file.endswith(".png"):
                    image_filepaths.append(os.path.join(root, file))

        images_to_gif(
            image_filepaths,
            gif_outpath=f"experiments/results/figures/{split}_propensity_score.gif",
            duration=2000,
        )

        for root, dirs, files in os.walk(image_directory):
            for file in files:
                os.remove(os.path.join(root, file))

    return model
