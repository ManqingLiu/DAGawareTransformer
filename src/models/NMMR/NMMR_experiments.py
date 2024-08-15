from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from argparse import ArgumentParser
import json
from pathlib import Path

from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch
import os

from src.dataset import (
    make_train_data,
    make_validation_data,
    make_test_data,
)

from src.models.NMMR.NMMR_trainers import NMMR_Trainer_DemandExperiment
from src.data.ate import generate_train_data_ate, generate_val_data_ate, generate_test_data_ate
from src.data.ate.data_class import PVTrainDataSetTorch, PVTestDataSetTorch


def NMMR_experiment(
    config: Dict[str, Any],
    data_config: Dict[str, Any],
    train_config_transformer: Dict[str, Any],
    model_config_transformer: Dict[str, Any],
    train_config_mlp: Dict[str, Any],
    model_config_mlp: Dict[str, Any],
    dag: Dict[str, Any],
    one_mdl_dump_dir: Path,
    random_seed: int = 42
):

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    train_data_mlp = generate_train_data_ate(data_config=data_config, rand_seed=random_seed)
    val_data_mlp = generate_val_data_ate(data_config=data_config, rand_seed=random_seed + 1)
    test_data_mlp = generate_test_data_ate(data_config=data_config)

    train_data_transformer = make_train_data(data_config, dag, random_seed)
    val_data_transformer = make_validation_data(data_config, dag, random_seed + 1)
    test_data_transformer, E_ydoA = make_test_data(data_config, val_data_transformer, dag)

    train_t = PVTrainDataSetTorch.from_numpy(train_data_mlp)
    val_data_t = PVTrainDataSetTorch.from_numpy(val_data_mlp)
    test_data_t = PVTestDataSetTorch.from_numpy(test_data_mlp)

    # mlp vs. transformer data: train data is the same, val data is different (TODO: why?)
    # test_data_mlp.treatment is (10, 1) while test_data_transformer.data['treatment'] is (10_000, 1)

    # collate_fn seems totally fine
    train_dataloader = DataLoader(
        train_data_transformer,
        batch_size=train_config_transformer["batch_size"],
        shuffle=False,
        collate_fn=train_data_transformer.collate_fn,
    )

    val_dataloader = DataLoader(
        val_data_transformer,
        batch_size=train_config_transformer["batch_size"],
        shuffle=False,
        collate_fn=val_data_transformer.collate_fn,
    )

    test_dataloader = DataLoader(
        test_data_transformer,
        batch_size=train_config_transformer["batch_size"],
        shuffle=False,
        collate_fn=test_data_transformer.collate_fn,
    )

    # __init__() just saves variables
    trainer = NMMR_Trainer_DemandExperiment(
        config,
        data_config,
        dag,
        train_config_transformer,
        model_config_transformer,
        train_config_mlp,
        model_config_mlp,
        random_seed,
        one_mdl_dump_dir,
    )

    model_transformer = trainer.train_transformer(train_dataloader, val_data_transformer, val_dataloader, test_dataloader)

    n_sample = data_config.get("n_sample", None)
    E_w_haw_transformer, oos_loss_transformer, _, _ = trainer.predict_transformer(
        model_transformer, data_config, val_data_transformer, dag, n_sample, test_dataloader
    )

    model_mlp = trainer.train_mlp(train_t, test_data_t, val_data_t)
    E_w_haw_mlp, oos_loss_mlp, _ = trainer.predict_mlp(
        model_mlp, data_config, test_data_t, val_data_t
    )
    
    return E_w_haw_transformer, E_w_haw_mlp, E_ydoA, oos_loss_transformer, oos_loss_mlp


if __name__ == "__main__":

    # Change the working directory to the project root
    # os.chdir(Path(__file__).resolve().parent.parent.parent.parent)
    parser = ArgumentParser()
    # Load the configurations from the JSON file
    with open(Path("config/train/proximal/nmmr_u_transformer_n1000.json"), "r") as f:
        config = json.load(f)

    # Extract the data and model configurations
    data_config = config["data"]
    train_config_transformer = config["training_transformer"]
    model_config_transformer = config["model_transformer"]
    train_config_mlp = config["training_mlp"]
    model_config_mlp = config["model_mlp"]

    # Define the directory where the model will be saved
    one_mdl_dump_dir = Path("experiments/results/proximal")

    args = parser.parse_args()

    with open(Path("config/dag/proximal_dag.json"), "r") as f:
        # print(f'Loading dag file from {args.dag}')
        dag = json.load(f)

    num_nodes = len(dag["nodes"])
    dag["node_ids"] = dict(zip(dag["nodes"], range(num_nodes)))
    mask = True

    # Run the experiment
    E_w_haw_transformer, E_w_haw_mlp, E_ydoA, oos_loss_transformer, oos_loss_mlp = NMMR_experiment(
        config,
        data_config,
        train_config_transformer,
        model_config_transformer,
        train_config_mlp,
        model_config_mlp,
        dag,
        one_mdl_dump_dir
    )

    # print E_ydoA
    print(f"E_ydoA: {E_ydoA}")
    # print E_w_haw
    print(f"E_w_haw_transformer: {E_w_haw_transformer}")
    print(f"E_w_haw_mlp: {E_w_haw_mlp}")
    # Print the oos_loss
    print(f"Out of sample loss transformer: {oos_loss_transformer}")
    print(f"Out of sample loss mlp: {oos_loss_mlp}")


