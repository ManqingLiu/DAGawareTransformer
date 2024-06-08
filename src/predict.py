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

from src.utils import *

from tqdm import tqdm


@staticmethod
def predict(model,
            data,
            dag,
            dataloader: DataLoader,
            mask: bool,
            random_seed: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = model.to(device)
    data = data[dag['nodes']]

    dataset = CausalDataset(data, dag, random_seed)
    model.eval()

    predictions = []
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
    prediction_transformer = PredictionTransformer(dataset.bin_edges)
    transformed_predictions = prediction_transformer.transform(predictions)

    return transformed_predictions






