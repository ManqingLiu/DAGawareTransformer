from argparse import ArgumentParser
import json
import pandas as pd
import numpy as np

import torch.nn as nn
from torch.utils.data import DataLoader
import torch

from src.model import *
from src.dataset import *

from typing import Dict
import time

from utils import *

from tqdm import tqdm


def predict(model: nn.Module,
            data,
            dag,
            dataloader: DataLoader,
            mask: bool,
            model_file: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = model.to(device)
    data = data[dag['nodes']]

    dataset = CausalDataset(data, dag)

    # Load the saved model weights
    model.load_state_dict(torch.load(model_file))
    model.eval()

    predictions = []
    attention_weights = []
    with torch.no_grad():
        for _, batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch, mask=mask)
            batch_predictions = []
            for output_name in outputs.keys():
                # Detach the outputs and move them to cpu
                output = outputs[output_name].cpu().numpy()
                output = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
                # Append the reshaped output to batch_predictions
                batch_predictions.append(output)
            # concatenate the batch predictions along the second axis
            batch_predictions = np.concatenate(batch_predictions, axis=1)
            predictions.append(batch_predictions)

    # assign column names to the predictions_df
    predictions = np.concatenate(predictions, axis=0)
    print(predictions.shape)
    print('Done predicting.')
    prediction_transformer = PredictionTransformer(dataset.bin_edges)
    transformed_predictions = prediction_transformer.transform(predictions)

    return transformed_predictions


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dag', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mask', type=bool, default=True)
    parser.add_argument('--data_train_file', type=str, required=True)
    parser.add_argument('--data_holdout_file', type=str, required=True)
    parser.add_argument('--model_train_file', type=str, required=True)
    parser.add_argument('--model_holdout_file', type=str, required=True)
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

    model = DAGTransformer(dag=dag,
                           **model_config)

    data = pd.read_csv(args.data_holdout_file)
    data = data[dag['nodes']]

    batch_size = train_config['batch_size']
    dataset = CausalDataset(data, dag)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=dataset.collate_fn)

    start_time = time.time()
    predictions_train = predict(model, data, dag, dataloader, mask=args.mask, model_file=args.model_train_file)



    data_A1 = replace_column_values(data, 't', 1)
    dataset_A1 = CausalDataset(data_A1, dag)
    dataloader_A1 = DataLoader(dataset_A1,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=dataset.collate_fn)

    predictions_train_A1 = predict(model, data_A1, dag, dataloader_A1, mask=args.mask, model_file=args.model_train_file)

    # rename pred_y to pred_y_A1
    predictions_train_A1 = predictions_train_A1.rename(columns={'pred_y': 'pred_y_A1'})

    data_A0 = replace_column_values(data, 't', 0)
    dataset_A0 = CausalDataset(data_A0, dag)
    dataloader_A0 = DataLoader(dataset_A0,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=dataset.collate_fn)
    predictions_train_A0 = predict(model, data_A0, dag, dataloader_A0, mask=args.mask, model_file=args.model_train_file)

    # rename pred_y to pred_y_A0
    predictions_train_A0 = predictions_train_A0.rename(columns={'pred_y': 'pred_y_A0'})

    # create final predictions train data including prob_t from predictions_train, pred_y_A1 from predictions_train_A1,
    # and pred_y_A0 from predictions_train_A0
    final_predictions_train = pd.concat([predictions_train['t_prob'],
                                         predictions_train_A1['pred_y_A1'],
                                         predictions_train_A0['pred_y_A0']], axis=1)

    # create final predictions holdout data
    data = pd.read_csv(args.data_holdout_file)
    data = data[dag['nodes']]
    dataset = CausalDataset(data, dag)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=dataset.collate_fn)
    predictions_holdout = predict(model, data, dag, dataloader, mask=args.mask, model_file=args.model_holdout_file)

    data_A1 = replace_column_values(data, 't', 1)
    dataset_A1 = CausalDataset(data_A1, dag)
    dataloader_A1 = DataLoader(dataset_A1,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=dataset.collate_fn)

    predictions_holdout_A1 = predict(model, data_A1, dag, dataloader_A1, mask=args.mask, model_file=args.model_holdout_file)

    # rename pred_y to pred_y_A1
    predictions_holdout_A1 = predictions_holdout_A1.rename(columns={'pred_y': 'pred_y_A1'})

    data_A0 = replace_column_values(data, 't', 0)
    dataset_A0 = CausalDataset(data_A0, dag)
    dataloader_A0 = DataLoader(dataset_A0,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=dataset.collate_fn)
    predictions_holdout_A0 = predict(model, data_A0, dag, dataloader_A0, mask=args.mask, model_file=args.model_holdout_file)

    # rename pred_y to pred_y_A0
    predictions_holdout_A0 = predictions_holdout_A0.rename(columns={'pred_y': 'pred_y_A0'})

    # create final predictions holdout data including prob_t from predictions_holdout, pred_y_A1 from predictions_holdout_A1,
    # and pred_y_A0 from predictions_holdout_A0
    final_predictions_holdout = pd.concat([predictions_holdout['t_prob'],
                                           predictions_holdout_A1['pred_y_A1'],
                                           predictions_holdout_A0['pred_y_A0']], axis=1)

    end_time = time.time()
    # Calculate and print the total wall time
    total_wall_time = end_time - start_time
    minutes, seconds = divmod(total_wall_time, 60)
    print(f"Total wall time used: {minutes} minutes and {seconds} seconds")
    transformed_predictions = pd.concat([final_predictions_train, final_predictions_holdout], axis=0)
    print(transformed_predictions.describe())

    # Save the predictions to a CSV file
    transformed_predictions.to_csv(args.output_file, index=False)




