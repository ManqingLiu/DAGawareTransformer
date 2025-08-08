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

# Import the Pure DAG Transformer implementation
from src.models.pure_dag_transformer import DAGTransformer
from src.models.dag_transformer_integration import train_pure_dag_transformer, predict_pure_dag_transformer


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
    if random_seed:
        torch.manual_seed(config["random_seed"])

    # Create Pure DAG Transformer model
    model = DAGTransformer(
        dag=dag,
        network_width=model_config["network_width"],
        embedding_dim=model_config["embedding_dim"],
        feedforward_dim=model_config["feedforward_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        dropout_rate=model_config["dropout_rate"],
        input_layer_depth=model_config["input_layer_depth"],
        encoder_weight=model_config["encoder_weight"],
        activation=model_config.get("activation", "relu"),
        use_layernorm=model_config.get("use_layernorm", False),
        name=f"dag_transformer_{estimator}"
    )
    # Train the model
    model, _, _ = train_pure_dag_transformer(
        data_name=data_name,
        estimator=estimator,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        val_data=val_data,
        pseudo_ate_data=pseudo_ate_data,
        sample_id=sample_id,
        config=config,
        dag=dag,
        random_seed=random_seed
    )

    # Generate predictions on test data
    predictions_test, metrics_test = predict_pure_dag_transformer(
        model=model,
        data_name=data_name,
        data=test_data,
        pseudo_ate_data=pseudo_ate_data,
        dag=dag,
        train_config=train_config,
        random_seed=random_seed,
        sample_id=sample_id,
        prefix="Test",
        estimator=estimator
    )

    return model, predictions_test, metrics_test


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--estimator", type=str, required=True)
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

    # Run experiment
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

    # Print metrics
    print(metrics_test)

    # Save predictions as CSV
    predictions_test.to_csv(filepaths[f"predictions_{args.estimator}"], index=False)

    # Calculate and print total wall time
    end_time = time.time()
    total_wall_time = end_time - start_time
    minutes, seconds = divmod(total_wall_time, 60)
    print(f"Total wall time used: {minutes:.0f} minutes and {seconds:.2f} seconds")