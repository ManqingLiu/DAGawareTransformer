from argparse import ArgumentParser
import json
from typing import Dict
import time

import pandas as pd
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import wandb

from src.data.utils_data import data_preprocess
from src.models.transformer_model import DAGTransformer


def experiment(data_name: str,
               estimator: str,
               config: Dict,
               dag: Dict,
               train_dataloader: DataLoader,
               val_dataloader: DataLoader,
               val_data: pd.DataFrame,
               pseudo_ate_data: pd.DataFrame,
               sample_id: int,
               random_seed=False,
               test_data: pd.DataFrame = None
               ):
    model_config = config[estimator]["model"]
    train_config = config[estimator]["training"]
    if random_seed == True:
        torch.manual_seed(config["random_seed"])

    wandb.init(project="DAG transformer", entity="mliu7", config=config)

    model = DAGTransformer(dag=dag, **model_config)
    model, rmse_cfcv, rmse_ipw = model._train(
        data_name,
        estimator,
        model,
        train_dataloader,
        val_dataloader,
        val_data,
        pseudo_ate_data,
        sample_id,
        config,
        dag,
        imbalance_loss_weight=train_config["imbalance_loss_weight"],
        random_seed=random_seed
    )

    predictions_test, metrics_val, metrics_test = model.predict(
        model,
        data_name,
        test_data,
        pseudo_ate_data,
        sample_id,
        dag,
        train_config,
        random_seed=random_seed
    )

    wandb.finish()

    return (model,
            rmse_cfcv,
            rmse_ipw,
            predictions_test,
            metrics_test)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True
    )
    parser.add_argument("--estimator", type=str, required=True
                        )
    parser.add_argument("--data_name", type=str, required=True)

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    filepaths = config["filepaths"]
    config_train = config[args.estimator]["training"]

    with open(filepaths["dag"]) as f:
        dag = json.load(f)

    (train_data, train_dataloader, val_data, val_dataloader, test_data, test_dataloader) = data_preprocess(
        args.estimator, config, filepaths, dag
    )

    start_time = time.time()

    if "lalonde" in args.data_name:
        pseudo_ate_data = pd.read_csv(filepaths["pseudo_ate_file"])
    else:
        pseudo_ate_data = pd.read_csv(filepaths["pseudo_cate_file"])

    sample_id = config["sample_id"]

    (model, rmse_cfcv, rmse_ipw, predictions_test, metrics_test) = experiment(
                                                                      args.data_name,
                                                                      args.estimator,
                                                                      config,
                                                                      dag,
                                                                      train_dataloader,
                                                                      val_dataloader,
                                                                      val_data,
                                                                      pseudo_ate_data,
                                                                      sample_id,
                                                                      random_seed=True,
                                                                      test_data=test_data)

    output_dir = filepaths[f"result_file_{args.estimator}"]
    print(metrics_test)

    # Convert all float32 values to native Python float
    for key, value in metrics_test.items():
        if isinstance(value, np.float32):
            metrics_test[key] = float(value)

    auc = roc_auc_score(predictions_test['t'], predictions_test['t_prob'])
    print(f"The test AUC is: {auc}")

    # Write the dictionary to a JSON file
    with open(output_dir, 'w') as file:
        json.dump(metrics_test, file, indent=4)

    # After training for holdout file
    end_time = time.time()

    # Calculate and print the total wall time
    total_wall_time = end_time - start_time
    # Convert the total wall time to minutes and seconds
    minutes, seconds = divmod(total_wall_time, 60)
    print(f"Total wall time used: {minutes} minutes and {seconds} seconds")