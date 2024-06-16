import numpy as np
import math
import re
import pandas as pd
import matplotlib
import warnings
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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

def extract_number(filepath):
    match = re.search(r'_(\d+)_', filepath)
    if match:
        return int(match.group(1))
    return float('inf')  # Return a large number if no match found

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
def calculate_distribution_distance_for_single_feature(
    x, w, a, group_level, metric="abs_smd"
):
    """

    Args:
        x (pd.Series): A single feature to check balancing.
        a (pd.Series): Group assignment of each sample.
        w (pd.Series): sample weights for balancing between groups in `a`.
        group_level: Value from `a` in order to divide the sample into one vs. rest.
        metric (str | callable): Either a key from DISTRIBUTION_DISTANCE_METRICS or a metric with
            the signature weighted_distance(x, y, wx, wy) calculating distance between the weighted
            sample x and weighted sample y (weights by wx and wy respectively).

    Returns:
        float: weighted distance between the samples assigned to `group_level`
            and the rest of the samples.
    """
    if not callable(metric):
        metric = DISTRIBUTION_DISTANCE_METRICS[metric]
    cur_treated_mask = a == group_level
    x_treated = x.loc[cur_treated_mask]
    w_treated = w.loc[cur_treated_mask]
    x_untreated = x.loc[~cur_treated_mask]
    w_untreated = w.loc[~cur_treated_mask]
    distribution_distance = metric(x_treated, x_untreated, w_treated, w_untreated)
    return distribution_distance


# from causallib library: https://github.com/BiomedSciAI/causallib/blob/master/causallib/metrics/weight_metrics.py
def calculate_covariate_balance(X, a, w, metric="abs_smd"):
    """Calculate covariate balance table ("table 1")

    Args:
        X (pd.DataFrame): Covariates.
        a (pd.Series): Group assignment of each sample.
        w (pd.Series): sample weights for balancing between groups in `a`.
        metric (str | callable): Either a key from DISTRIBUTION_DISTANCE_METRICS or a metric with
            the signature weighted_distance(x, y, wx, wy) calculating distance between the weighted
            sample x and weighted sample y (weights by wx and wy respectively).

    Returns:
        pd.DataFrame: index are covariate names (columns) from X, and columns are
            "weighted" / "unweighted" results of applying `metric` on each covariate
            to compare the two groups.
    """
    treatment_values = np.sort(np.unique(a))
    results = {}
    for treatment_value in treatment_values:
        distribution_distance_of_cur_treatment = pd.DataFrame(
            index=X.columns, columns=["weighted", "unweighted"], dtype=float
        )
        for col_name, col_data in X.items():
            weighted_distance = calculate_distribution_distance_for_single_feature(
                col_data, w, a, treatment_value, metric
            )
            unweighted_distance = calculate_distribution_distance_for_single_feature(
                col_data, pd.Series(1, index=w.index), a, treatment_value, metric
            )
            distribution_distance_of_cur_treatment.loc[
                col_name, ["weighted", "unweighted"]
            ] = [weighted_distance, unweighted_distance]
        results[treatment_value] = distribution_distance_of_cur_treatment
    results = pd.concat(
        results, axis="columns", names=[a.name or "a", metric]
    )  # type: pd.DataFrame
    results.index.name = "covariate"
    if len(treatment_values) == 2:
        # If there are only two treatments, the results for both are identical.
        # Therefore, we can get rid of one of them.
        # We keep the results for the higher valued treatment group (assumed treated, typically 1):
        results = results.xs(treatment_values.max(), axis="columns", level=0)
    return results


def smd_plot(df, ax=None, epoch=None):
    """
    Plot the absolute standardized mean difference for weighted and unweighted data.

    Args:
        df (pd.DataFrame): DataFrame containing the 'weighted' and 'unweighted' columns.
        ax (plt.Axes | None): Matplotlib Axes to plot on. If None, use the current Axes.

    Returns:
        ax (plt.Axes): The Axes with the plot.
    """
    ax = ax or plt.gca()

    # Plot weighted and unweighted points
    ax.scatter(df['weighted'], df.index, label='weighted', color='blue')
    ax.scatter(df['unweighted'], df.index, label='unweighted', color='orange')

    # Connect weighted and unweighted points with lines
    for covariate in df.index:
        ax.plot([df.loc[covariate, 'weighted'], df.loc[covariate, 'unweighted']], [covariate, covariate], 'k--')

    # Adding labels and title
    ax.set_xlim([0, 2])
    ax.set_xlabel('Absolute Standardized Mean Difference')
    ax.set_ylabel('Covariates')
    ax.set_title(f'Number of epochs:{epoch}')
    ax.legend()

    return ax

# from causallib: https://causallib.readthedocs.io/en/latest/_modules/causallib/evaluation/plots/plots.html#plot_propensity_score_distribution
def plot_propensity_score_distribution(
    propensity,  
    treatment,
    num_bins: int,
    reflect=True,
    kde=False,
    cumulative=False,
    norm_hist=True,
    ax=None,
    epoch=None
):
    """
    Plot the distribution of propensity score

    Args:
        propensity (pd.Series):
        treatment (pd.Series):
        reflect (bool): Whether to plot second treatment group on the opposite sides of the x-axis.
                        This can only work if there are exactly two groups.
        kde (bool): Whether to plot kernel density estimation
        cumulative (bool): Whether to plot cumulative distribution.
        norm_hist (bool): If False - use raw counts on the y-axis.
                          If kde=True, then norm_hist should be True as well.
        ax (plt.Axes | None):

    Returns:

    """
    # assert propensity.index.symmetric_difference(a.index).size == 0
    ax = ax or plt.gca()
    if kde and not norm_hist:
        warnings.warn(
            "kde=True and norm_hist=False is not supported. Forcing norm_hist from False to True."
        )
        norm_hist = True
    plot_params = dict(bins=num_bins, density=norm_hist, alpha=0.5, cumulative=cumulative, range=(0, 1))

    unique_treatments = np.sort(np.unique(treatment))
    for treatment_number, treatment_value in enumerate(unique_treatments):
        cur_propensity = propensity.loc[treatment == treatment_value]
        cur_color = f"C{treatment_number}"
        ax.hist(
            cur_propensity,
            label=f"treatment = {treatment_value}",
            color=[cur_color],
            **plot_params,
        )
        
        ax.set_ylim([-100, 100])

        if kde:
            cur_kde = gaussian_kde(cur_propensity)
            min_support = max(0, cur_propensity.values.min() - cur_kde.factor)
            max_support = min(1, cur_propensity.values.max() + cur_kde.factor)
            X_plot = np.linspace(min_support, max_support, 200)
            if cumulative:
                density = np.array(
                    [cur_kde.integrate_box_1d(X_plot[0], x_i) for x_i in X_plot]
                )
                ax.plot(
                    X_plot,
                    density,
                    color=cur_color,
                )
            else:
                ax.plot(
                    X_plot,
                    cur_kde.pdf(X_plot),
                    color=cur_color,
                )
    if reflect:
        if len(unique_treatments) != 2:
            raise ValueError(
                "Reflecting density across X axis can only be done for two groups. "
                "This one has {}".format(len(unique_treatments))
            )
        # Update line:
        if kde:
            last_line = ax.get_lines()[-1]
            last_line.set_ydata(-1 * last_line.get_ydata())
        # Update histogram bars:
        idx_of_first_hist_rect = [patch.get_label() for patch in ax.patches].index(
            f"treatment = {unique_treatments[-1]}"
        )
        for patch in ax.patches[idx_of_first_hist_rect:]:
            patch.set_height(-1 * patch.get_height())

        ax.autoscale(enable=False, axis="both")
        # Remove negation sign from lower y-axis:
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda x, pos: str(x) if x >= 0 else str(-x)
            )
        )

    ax.legend(loc="best")
    x_type = (
        "Propensity" if propensity.between(0, 1, inclusive="both").all() else "Weights"
    )
    ax.set_xlabel(x_type)
    y_type = "Probability density" if norm_hist else "Counts"
    ax.set_ylabel(y_type)
    ax.set_title(f"{x_type} Distribution, epoch:{epoch}")
    return ax

def predict_function(model, dataset, dataloader, mask):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
