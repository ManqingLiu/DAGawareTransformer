import numpy as np
import math
import re
import pandas as pd
import matplotlib
import warnings
import matplotlib.pyplot as plt
import zipfile
from urllib.parse import urlparse
from scipy.stats import gaussian_kde

from argparse import ArgumentParser
import pandas as pd
from pandas import DataFrame as pdDataFrame, Series as pdSeries
import numpy as np
import requests

import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import json
import hashlib
import time
import os

from src.model import DAGTransformer, causal_loss_fun
from src.dataset import *
from src.predict import *

def extract_number(filepath):
    match = re.search(r'_(\d+)_', filepath)
    if match:
        return int(match.group(1))
    return float('inf')  # Return a large number if no match found


# stratified standardization estimator
def standardization(outcome_hat_A1, outcome_hat_A0):
    '''
    Args:
        outcome_hat_A1: prediction of outcome had all received treatment A=1
        outcome_hat_A0: prediction of outcome had all received treatment A=0

    Returns:
        tau_hat: mean of the difference between outcome_hat_A1 and outcome_hat_A0
    '''

    tau_hat = np.mean(outcome_hat_A1)-np.mean(outcome_hat_A0)

    return tau_hat



def rmse(value1, value2):
    squared_difference = np.mean((value1 - value2) ** 2)
    root_mean_square_error = np.sqrt(squared_difference)
    return root_mean_square_error

def relative_bias(true_value, estimated_value):
    bias = (estimated_value - true_value) / true_value
    return bias


def replace_column_values(df: pd.DataFrame, column: str, new_value: int) -> pd.DataFrame:
    new_df = df.copy()
    new_df[column] = new_value
    return new_df

def detect_variable_type(df, col):
    # Detect if the variable is continuous or binary
    unique_values = df[col].unique()
    if len(unique_values) <= 2:
        return 'binary'
    else:
        return 'continuous'


def plot_attention_heatmap(attention_weights, layer_idx=0, head_idx=0, node_names=None, transpose=False):
    attn_matrix = attention_weights[layer_idx][head_idx].detach().numpy()  # Ensure it's detached and on CPU
    if transpose:
        attn_matrix = attn_matrix.T  # Transpose the matrix

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn_matrix, cmap='viridis')

    # Set the number of ticks based on the attention matrix size
    ax.set_xticks(np.arange(len(attn_matrix[0])))  # Adjust based on number of columns
    ax.set_yticks(np.arange(len(attn_matrix)))  # Adjust based on number of rows

    # Setting node names as tick labels if provided
    if node_names:
        ax.set_xticklabels(node_names)
        ax.set_yticklabels(node_names)
    else:
        ax.set_xticklabels(range(1, len(attn_matrix[0]) + 1))
        ax.set_yticklabels(range(1, len(attn_matrix) + 1))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Create text annotations inside the heatmap
    for i in range(len(attn_matrix)):
        for j in range(len(attn_matrix[0])):
            text = ax.text(j, i, f"{attn_matrix[i, j]:.2f}",
                           ha="center", va="center", color="w")

    ax.set_title("Attention Weights Heatmap")
    fig.tight_layout()
    plt.colorbar(im)


def log_results(estimator: str,
                orig_config_path: str,
                new_config: Dict[str, Any]) -> Dict[str, Any]:
    # Load the original config from the JSON file
    with open(orig_config_path, 'r') as file:
        orig_config = json.load(file)

    # Ensure orig_config is a dictionary
    if not isinstance(orig_config, dict):
        raise TypeError("The original config file does not contain a dictionary")

    # Replace the "training" and "model" parts of the config
    orig_config[estimator]["training"] = new_config[estimator]["training"]
    orig_config[estimator]["model"] = new_config[estimator]["model"]

    # Optionally, save the updated config back to the JSON file
    with open(orig_config_path, 'w') as file:
        json.dump(orig_config, file, indent=4)

def log_results_proximal(filepaths, config):
    # Write everything back to the file
    with open(filepaths, 'w') as f:
        json.dump(config, f, indent=4)

def log_results_evaluate(results, config, results_file):
    # Check if the file exists
    if os.path.isfile(results_file):
        # If the file exists, load the existing data
        with open(results_file, 'r') as f:
            existing_data = json.load(f)
    else:
        # If the file does not exist, initialize an empty list to store data
        existing_data = []

    # Append the new results and config
    existing_data.append({"results": results, "config": config})

    # Write everything back to the file
    with open(results_file, 'w') as f:
        json.dump(existing_data, f, indent=4)

# from causallib library:https://github.com/BiomedSciAI/causallib/blob/master/causallib/utils/stat_utils.py#L181
def calc_weighted_standardized_mean_differences(x, y, wx, wy, weighted_var=False):
    r"""
    Standardized mean difference: frac{\mu_1 - \mu_2 }{\sqrt{\sigma_1^2 + \sigma_2^2}}

    References:
        [1]https://cran.r-project.org/web/packages/cobalt/vignettes/cobalt_A0_basic_use.html#details-on-calculations
        [2]https://en.wikipedia.org/wiki/Strictly_standardized_mean_difference#Concept

    Note on variance:
    - The variance is calculated on unadjusted to avoid paradoxical situation when adjustment decreases both the
      mean difference and the spread of the sample, yielding a larger smd than that prior to adjustment,
      even though the adjusted groups are now more similar [1].
    - The denominator is as depicted in the "statistical estimation" section:
      https://en.wikipedia.org/wiki/Strictly_standardized_mean_difference#Statistical_estimation,
      namely, disregarding the covariance term [2], and is unweighted as suggested above in [1].
    """
    numerator = np.average(x, weights=wx) - np.average(y, weights=wy)
    if weighted_var:
        var = lambda vec, weights: np.average((vec - np.average(vec, weights=weights)) ** 2, weights=weights)
        denominator = np.sqrt(var(x, wx) + var(y, wy))
    else:
        denominator = np.sqrt(np.nanvar(x) + np.nanvar(y))
    if np.isfinite(denominator) and np.isfinite(numerator) and denominator != 0:
        bias = numerator / denominator
    else:
        bias = np.nan
    return bias



# from causallib library: https://github.com/BiomedSciAI/causallib/blob/master/causallib/metrics/weight_metrics.py

DISTRIBUTION_DISTANCE_METRICS = {
    "smd": lambda x, y, wx, wy: calc_weighted_standardized_mean_differences(
        x, y, wx, wy
    ),
    "abs_smd": lambda x, y, wx, wy: abs(
        calc_weighted_standardized_mean_differences(x, y, wx, wy)
    ),
}
def predict_function(model, train_config, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize an empty dictionary to store results
    all_results = {output_name: [] for output_name in model.output_nodes.keys()}
    with torch.no_grad():
        for batch_raw, batch_binned in dataloader:
            batch = {k: v.to(device) for k, v in batch_raw.items()}
            outputs = model(batch, mask=train_config["dag_attention_mask"])
            for output_name in all_results:
                all_results[output_name].append(outputs[output_name].cpu())

    # Concatenate results for each output name across all batches
    for output_name in all_results:
        if all_results[output_name]:  # Check if the list is not empty
            all_results[output_name] = torch.cat(all_results[output_name], dim=0)
        else:
            all_results[output_name] = torch.tensor([])
    return all_results


DATA_FOLDER = 'data/ihdp'

def download_file(url, file_path):
    # open in binary mode
    with open(file_path, "wb") as f:
        # get request
        response = requests.get(url)
        # write to file
        f.write(response.content)
def download_dataset(url, dataset_name, dataroot=None, filename=None):
    if dataroot is None:
        dataroot = DATA_FOLDER
    if filename is None:
        filename = os.path.basename(urlparse(url).path)
    file_path = os.path.join(dataroot, filename)
    if os.path.isfile(file_path):
        print('{} dataset already exists at {}'.format(dataset_name, file_path))
    else:
        print('Downloading {} dataset to {} ...'.format(dataset_name, file_path), end=' ')
        download_file(url, file_path)
        print('DONE')
    return file_path

def unzip(path_to_zip_file, unzip_dir=None):
    unzip_path = os.path.splitext(path_to_zip_file)[0]
    if os.path.isfile(unzip_path):
        print('File already unzipped at', unzip_path)
        return unzip_path

    print('Unzipping {} to {} ...'.format(path_to_zip_file, unzip_path), end=' ')
    if unzip_dir is None:
        unzip_dir = os.path.dirname(path_to_zip_file)
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)
    print('DONE')
    return unzip_path


def robust_lookup(df, indexer):
    """
    Robust way to apply pandas lookup when indices are not unique

    Args:
        df (pdDataFrame):
        indexer (pdSeries): A Series whose index is either same or a subset of `df.index`
                            and whose values are values from `df.columns`.
                            If `a.index` contains values not in `df.index`
                            they will have NaN values.

    Returns:
        pdSeries: a vector where (logically) `extracted[i] = df.loc[indexer.index[i], indexer[i]]`.
            In most cases, when `indexer.index == df.index` this translates to
            `extracted[i] = df.loc[i, indexer[i]]`
    """
    # Convert the index into
    idx, col = indexer.factorize()  # convert text labels into integers
    extracted = df.reindex(col, axis=1).reindex(indexer.index, axis=0)  # make sure the columns exist and the indeces are the same
    extracted = extracted.to_numpy()[range(len(idx)), idx]  # numpy accesses by location, not by named index
    extracted = pdSeries(extracted, index=indexer.index)
    return extracted