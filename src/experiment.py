from argparse import ArgumentParser
import json
import os
import time
from typing import Dict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from src.dataset import CausalDataset, PredictionTransformer
from src.models.transformer_model import DAGTransformer, causal_loss_fun
from src.predict import predict
from src.train.lalonde_psid.train import train
from utils import IPTW_unstabilized, rmse, replace_column_values





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dag', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mask', type=bool, default=True)
    parser.add_argument('--data_train_file', type=str, required=True)
    parser.add_argument('--data_val_file', type=str, required=True)
    parser.add_argument('--data_test_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    with open(args.dag) as f:
        print(f'Loading dag file from {args.dag}')
        dag = json.load(f)

    with open(args.config) as f:
        print(f'Loading config file from {args.config}')
        config = json.load(f)

    train_config = config['training']
    model_config = config['model']

    # Move all of this to a utils function for load_dag and load_data
    num_nodes = len(dag['nodes'])
    dag['node_ids'] = dict(zip(dag['nodes'], range(num_nodes)))
    print(dag)

    model = DAGTransformer(dag=dag, **model_config)

    train_data = pd.read_csv(args.data_train_file)
    train_data_model = train_data[dag['nodes']]
    train_dataset = CausalDataset(train_data_model, dag)

    batch_size = train_config['batch_size']
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=train_dataset.collate_fn)

    val_data = pd.read_csv(args.data_val_file)
    val_data_model = val_data[dag['nodes']]
    val_dataset = CausalDataset(val_data_model, dag)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=val_dataset.collate_fn)

    test_data = pd.read_csv(args.data_test_file)
    test_data_model = test_data[dag['nodes']]
    test_dataset = CausalDataset(test_data_model, dag)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 collate_fn=test_dataset.collate_fn)

    # Before training for train file
    start_time = time.time()
    model = train(model,
                  val_data,
                  train_data,
                  dag,
                  train_dataloader,
                  val_dataloader,
                  test_dataloader,
                  config,
                  mask=args.mask)
    print('Done training.')


    predictions = predict(model,val_data, dag, val_dataloader, mask=args.mask)

    data_A1 = replace_column_values(val_data, 't', 1)
    dataset_A1 = CausalDataset(data_A1, dag)
    dataloader_A1 = DataLoader(dataset_A1,
                               batch_size=batch_size,
                               shuffle=True,
                               collate_fn=val_dataset.collate_fn)

    predictions_A1 = predict(model, data_A1, dag, dataloader_A1, mask=args.mask)

    # rename pred_y to pred_y_A1
    predictions_A1 = predictions_A1.rename(columns={'pred_y': 'pred_y_A1'})

    data_A0 = replace_column_values(val_data, 't', 0)
    dataset_A0 = CausalDataset(data_A0, dag)
    dataloader_A0 = DataLoader(dataset_A0,
                               batch_size=batch_size,
                               shuffle=True,
                               collate_fn=val_dataset.collate_fn)
    predictions_A0 = predict(model, data_A0, dag, dataloader_A0, mask=args.mask)

    # rename pred_y to pred_y_A0
    predictions_A0 = predictions_A0.rename(columns={'pred_y': 'pred_y_A0'})

    # create final predictions train data including prob_t from predictions_train, pred_y_A1 from predictions_train_A1,
    # and pred_y_A0 from predictions_train_A0
    final_predictions = pd.concat([predictions['t_prob'],
                                   predictions_A1['pred_y_A1'],
                                   predictions_A0['pred_y_A0']], axis=1)

    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the predictions to a CSV file
    final_predictions.to_csv(args.output_file, index=False)

    # After training for holdout file
    end_time = time.time()

    # Calculate and print the total wall time
    total_wall_time = end_time - start_time
    # Convert the total wall time to minutes and seconds
    minutes, seconds = divmod(total_wall_time, 60)
    print(f"Total wall time used: {minutes} minutes and {seconds} seconds")