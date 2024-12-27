import json
from typing import Dict
from argparse import ArgumentParser
import pandas as pd

from time import time

import torch
from torch.utils.data import DataLoader
from ray import train, tune

from experiments.tuning.config import config_lalonde_cps, config_acic
from src.utils import log_results
from src.data.utils_data import data_preprocess
from src.experiment import experiment

def fine_tune(config: Dict,
              data_name: str,
              estimator: str,
              dag: Dict,
              pseudo_ate_data: pd.DataFrame,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              val_data: pd.DataFrame,
              test_data: pd.DataFrame,
              sample_id: int):
    """
    Fine-tune the model with the given configuration and data.

    Args:
        config (dict): Configuration dictionary containing training parameters.
        estimator (str): Estimator to fine-tune.
        dag (object): Directed Acyclic Graph object.
        pseudo_ate_data (DataFrame): Pseudo average treatment effect data.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        val_data (DataFrame): Validation dataset.
        sample_id (int): Sample ID for tracking.

    Returns:
        None
    """
    start_time = time()

    # Run the experiment
    model, rmse_cfcv, rmse_ipw, predictions_test, metrics_test = experiment(
        data_name=data_name,
        estimator=estimator,
        config=config,
        dag=dag,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        val_data=val_data,
        pseudo_ate_data=pseudo_ate_data,
        sample_id=sample_id,
        random_seed=True,
        test_data=test_data
    )

    # Report metrics
    train.report({
        "cfcv_rmse":rmse_cfcv,
        "ipw_rmse": rmse_ipw
    })

    # Check if the time limit is reached
    elapsed_time = time() - start_time
    if elapsed_time >= config[estimator]['training']['time_limit']:
        print(f"Time limit of {config['training']['time_limit']} seconds reached. Stopping fine-tuning.")
        return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True
    )
    parser.add_argument("--estimator", type=str, required=True
                        )
    parser.add_argument("--data_name", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    filepaths = config["filepaths"]

    with open(filepaths["dag"]) as f:
        dag = json.load(f)

    if "lalonde" in args.data_name:
        pseudo_ate_data = pd.read_csv(filepaths["pseudo_ate_file"])
    else:
        pseudo_ate_data = pd.read_csv(filepaths["pseudo_cate_file"])


    (train_data, train_dataloader, val_data,
     val_dataloader, test_data, test_dataloader) = data_preprocess(
        args.estimator, config, filepaths, dag
    )

    fine_tune_new = tune.with_parameters(fine_tune,
                                         data_name=args.data_name,
                                         estimator=args.estimator,
                                         dag=dag,
                                         pseudo_ate_data=pseudo_ate_data,
                                         train_dataloader=train_dataloader,
                                         val_dataloader=val_dataloader,
                                         val_data=val_data,
                                         test_data=test_data,
                                         sample_id=config["sample_id"])

    # Step 1: Define a dictionary mapping data names to configurations
    configurations = {
        "lalonde_cps": config_lalonde_cps,
        "acic": config_acic
    }

    # Step 2: Select the correct configuration using args.data_name
    selected_config = configurations.get(args.data_name)

    analysis = tune.run(
        fine_tune_new,
        config=selected_config,
        num_samples=20,  # Number of times to sample from the hyperparameter space
        resources_per_trial={"cpu": 10, "gpu": 2 if torch.cuda.is_available() else 0},
        metric=f"{args.estimator}_rmse",
        mode="min",
        time_budget_s=14400
    )

    best_trial = analysis.get_best_trial(f"{args.estimator}_rmse", "min", "all")

    print(f"Best trial config {args.estimator}: {best_trial.config}")
    print(f"Best trial final validation {args.estimator} RMSE: {best_trial.last_result[f'{args.estimator}_rmse']}")

    # Log the results
    log_results(args.estimator,
                filepaths["config"],
                best_trial.config)