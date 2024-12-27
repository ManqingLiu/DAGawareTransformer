from argparse import ArgumentParser
import json
from typing import Dict
import time

import pandas as pd
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data.utils_data import data_preprocess
from src.models.dag_aware_transformer import DAGTransformer


def experiment(data_name: str,
               estimator: str,
               config: Dict,
               dag: Dict,
               train_dataloader: DataLoader,
               val_dataloader: DataLoader,
               val_data: pd.DataFrame,
               pseudo_ate_data: pd.DataFrame,
               random_seed=False,
               sample_id: int = None,
               test_data: pd.DataFrame = None
               ):
    model_config = config[estimator]["model"]
    train_config = config[estimator]["training"]
    if random_seed == True:
        torch.manual_seed(config["random_seed"])

    model = DAGTransformer(dag=dag, **model_config)
    model, _, _ = model._train(
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
        random_seed=random_seed
    )

    predictions_test, metrics_test = model.predict(
        model,
        data_name,
        test_data,
        pseudo_ate_data,
        dag,
        train_config,
        random_seed=random_seed,
        sample_id = sample_id,
        prefix="Test",
        estimator=estimator
    )

    return (model, predictions_test, metrics_test)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True
    )
    parser.add_argument("--estimator", type=str, required=True
                        )
    parser.add_argument("--dag", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--sample_id", type=int, required=False, default=None)

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    filepaths = config["filepaths"]
    config_train = config[args.estimator]["training"]

    with open(filepaths[args.dag]) as f:
        dag = json.load(f)

    (train_data, train_dataloader, val_data, val_dataloader, test_data, test_dataloader) = data_preprocess(
        args.estimator, config, filepaths, dag
    )

    start_time = time.time()

    if "lalonde" in args.data_name:
        pseudo_ate_data = pd.read_csv(filepaths["pseudo_ate_file"])
    else:
        pseudo_ate_data = pd.read_csv(filepaths["pseudo_cate_file"])


    (model, predictions_test, metrics_test) = experiment(
                                      args.data_name,
                                      args.estimator,
                                      config,
                                      dag,
                                      train_dataloader,
                                      test_dataloader,
                                      test_data,
                                      pseudo_ate_data,
                                      sample_id=args.sample_id,
                                      random_seed=True,
                                      test_data=test_data)

    # print metircs_test
    print(metrics_test)

    # save predictions as csv
    predictions_test.to_csv(filepaths[f"predictions_{args.estimator}"], index=False)

    # After training for holdout file
    end_time = time.time()

    # Calculate and print the total wall time
    total_wall_time = end_time - start_time
    # Convert the total wall time to minutes and seconds
    minutes, seconds = divmod(total_wall_time, 60)
    print(f"Total wall time used: {minutes} minutes and {seconds} seconds")