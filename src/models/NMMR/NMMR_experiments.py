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


def NMMR_experiment(
    config: Dict[str, Any],
    data_config: Dict[str, Any],
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    dag: Dict[str, Any],
    mask: bool,
    one_mdl_dump_dir: Path,
    random_seed: int = 42,
    verbose: int = 0,
):

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    train_data = make_train_data(data_config, dag, random_seed)
    val_data = make_validation_data(data_config, dag, random_seed + 1)
    test_data, E_ydoA = make_test_data(data_config, val_data, dag)

    train_dataloader = DataLoader(
        train_data,
        batch_size=train_config["batch_size"],
        shuffle=True,
        collate_fn=train_data.collate_fn,
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=train_config["batch_size"],
        shuffle=True,
        collate_fn=val_data.collate_fn,
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=train_config["batch_size"],
        shuffle=False,
        collate_fn=test_data.collate_fn,
    )

    trainer = NMMR_Trainer_DemandExperiment(
        config,
        data_config,
        dag,
        train_config,
        model_config,
        mask,
        random_seed,
        one_mdl_dump_dir,
    )

    trainer.train(train_dataloader, val_dataloader, verbose)

    if trainer.gpu_flg:
        torch.cuda.empty_cache()
        test_data = test_data.to_gpu()

    n_sample = data_config.get("n_sample", None)
    E_wx_hawx = trainer.predict(
        n_sample, test_dataloader
    )

    pred = E_wx_hawx
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    np.savetxt(one_mdl_dump_dir.joinpath(f"{timestamp}_{random_seed}.pred.txt"), pred)

    # test_data.structural is equivalent to EY_doA
    np.testing.assert_array_equal(pred.shape, E_ydoA.shape)
    oos_loss = np.mean((pred - E_ydoA) ** 2)
    
    return oos_loss


if __name__ == "__main__":

    # Change the working directory to the project root
    os.chdir(Path(__file__).resolve().parent.parent.parent.parent)
    parser = ArgumentParser()
    # Load the configurations from the JSON file
    with open(Path("config/train/proximal/nmmr_u_transformer_n5000.json"), "r") as f:
        config = json.load(f)

    # Extract the data and model configurations
    data_config = config["data"]
    train_config = config["training"]
    model_config = config["model"]

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
    oos_loss = NMMR_experiment(
        config, data_config, model_config, train_config, dag, mask, one_mdl_dump_dir
    )
    # Print the oos_loss
    print(f"Out of sample loss: {oos_loss}")
