import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

from argparse import ArgumentParser
import pandas as pd
import numpy as np

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

def IPTW_stabilized(t, y, pred_t):
    """
    Calculate the treatment effect using Stabilized Inverse Probability of Treatment Weighting (IPTW) without trimming.

    Parameters:
    - t (array-like): Treatment indicator variable (1 for treated, 0 for untreated).
    - y (array-like): Outcome variable.
    - pred_t (array-like): Predicted propensity score, e(w), for receiving treatment.

    Returns:
    - tau_hat (float): Estimated treatment effect, stabilized.
    """

    # Ensure inputs are numpy arrays for element-wise operations
    #t = np.array(t)
    #y = np.array(y)
    #pred_t = np.array(pred_t)

    # Calculate the proportion of treated and untreated
    pt_1 = np.mean(t)
    pt_0 = 1 - pt_1

    # Calculate stabilized weights without applying trimming
    weights_treated = pt_1 / pred_t
    weights_untreated = pt_0 / (1 - pred_t)

    # Calculate the numerator for each observation
    numerator = (t * y * weights_treated) - ((1 - t) * y * weights_untreated)


    # Sum over all observations and divide by the number of observations to get tau_hat
    tau_hat = np.sum(numerator) / len(t)

    return tau_hat

def IPTW_unstabilized(t, y, pred_t):
    """
    Calculate the treatment effect using Stabilized Inverse Probability of Treatment Weighting (IPTW) without trimming.

    Parameters:
    - t (array-like): Treatment indicator variable (1 for treated, 0 for untreated).
    - y (array-like): Outcome variable.
    - pred_t (array-like): Predicted propensity score, e(w), for receiving treatment.

    Returns:
    - tau_hat (float): Estimated treatment effect, stabilized.
    """

    # Ensure inputs are numpy arrays for element-wise operations
    #t = np.array(t)
    #y = np.array(y)
    #pred_t = np.array(pred_t)

    # Calculate stabilized weights without applying trimming
    weights_treated = 1 / pred_t
    weights_untreated = 1 / (1 - pred_t)

    # Calculate the numerator for each observation
    numerator = (t * y * weights_treated) - ((1 - t) * y * weights_untreated)


    # Sum over all observations and divide by the number of observations to get tau_hat
    tau_hat = np.sum(numerator) / len(t)

    return tau_hat

def AIPW_Y1(t, pred_t, y, pred_y):
    """
    Calculate the counterfacutal outcome of Y^{a=1} using Augmented Inverse Probability of Treatment Weighting (AIPW).

    Parameters:
    - t (array-like): Treatment indicator variable (1 for treated, 0 for untreated).
    - y (array-like): Outcome variable.
    - pred_t (array-like): Predicted propensity score, e(w), for receiving treatment.
    - pred_y (array-like): Predicted outcome variable.

    Returns:
    - tau_hat (float): Estimated treatment effect.
    """

    # Calculate stabilized weights without applying trimming
    weights_treated = 1 / pred_t
    #weights_untreated = 1 / (1 - pred_t)


    y1 = pred_y + (t*weights_treated)*(y-pred_y)
    #y0 = pred_y + ((1-t)*weights_untreated)*(y-pred_y)

    # Calculate the AIPW estimate
    Y1 = np.mean(y1)

    return Y1

def AIPW_Y0(t, pred_t, y, pred_y):
    """
    Calculate the counterfactual outcome of Y^{a=0} using Augmented Inverse Probability of Treatment Weighting (AIPW).

    Parameters:
    - t (array-like): Treatment indicator variable (1 for treated, 0 for untreated).
    - y (array-like): Outcome variable.
    - pred_t (array-like): Predicted propensity score, e(w), for receiving treatment.
    - pred_y (array-like): Predicted outcome variable.

    Returns:
    - tau_hat (float): Estimated treatment effect.
    """

    # Calculate stabilized weights without applying trimming
    weights_untreated = 1 / (1 - pred_t)

    y0 = pred_y + ((1-t)*weights_untreated)*(y-pred_y)

    # Calculate the AIPW estimate
    Y0 = np.mean(y0)

    return Y0


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
    squared_difference = (value1 - value2) ** 2
    root_mean_square_error = math.sqrt(squared_difference)
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


def log_results(config, results_file):
    # Extract the "training" and "model" parts of the config
    training_config = config["training"]
    model_config = config["model"]

    # Log the "training" and "model" configs
    with open(results_file, 'w') as f:
        json.dump({"training": training_config, "model": model_config}, f)

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