from argparse import ArgumentParser
import json
import os
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
import wandb

from src.data.data_preprocess import data_preprocess
from src.models.transformer_model import DAGTransformer


def experiment(config, dag, train_dataloader, val_dataloader, val_data, random_seed=False):
    model_config = config["model"]
    train_config = config["training"]
    if random_seed == True:
        torch.manual_seed(config["random_seed"])

    wandb.init(project="DAG transformer", entity="mliu7", config=config)

    model = DAGTransformer(dag=dag, **model_config)
    model._train(
        model,
        train_dataloader,
        val_dataloader,
        train_config,
        imbalance_loss_weight=train_config["imbalance_loss_weight"]
    )

    predictions = model.predict(
        model,
        val_data,
        dag,
        train_config,
        mask=train_config["dag_attention_mask"],
        random_seed=random_seed
    )

    wandb.finish()

    return predictions

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, default="config/train/lalonde_psid.json"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    filepaths = config["filepaths"]

    with open(filepaths["dag"]) as f:
        dag = json.load(f)

    (train_data, train_dataloader, val_data, val_dataloader, test_data, test_dataloader) = data_preprocess(
        config, filepaths, dag
    )

    start_time = time.time()

    predictions = experiment(config, dag, train_dataloader, val_dataloader, val_data, random_seed=True)

    output_dir = os.path.dirname(filepaths["output_file"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the predictions to a CSV file
    predictions.to_csv(filepaths["output_file"], index=False)

    # summary mean of pred_y_A1 and pred_y_A0 in predictions
    summary = predictions[["pred_y_A0", "pred_y_A1"]].mean()
    print(summary)

    # After training for holdout file
    end_time = time.time()

    # Calculate and print the total wall time
    total_wall_time = end_time - start_time
    # Convert the total wall time to minutes and seconds
    minutes, seconds = divmod(total_wall_time, 60)
    print(f"Total wall time used: {minutes} minutes and {seconds} seconds")
