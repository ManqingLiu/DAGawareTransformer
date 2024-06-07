from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import ks_2samp

from src.dataset import CausalDataset, PredictionTransformer
from utils import (IPTW_unstabilized, rmse, calculate_covariate_balance,
                   smd_plot, plot_propensity_score_distribution, extract_number)


def calculate_metrics(
    dataset: pd.DataFrame, dag: Dict, predictions: np.array, prefix: str
) -> Dict[str, float]:
    # assign column names to the predictions_df
    causal_dataset = CausalDataset(dataset, dag)
    prediction_transformer = PredictionTransformer(causal_dataset.bin_edges)
    transformed_predictions = prediction_transformer.transform(predictions)

    predictions_final = pd.concat([dataset, transformed_predictions], axis=1)
    predictions_final["weight"] = np.where(
        predictions_final["t"] == 1,
        1 / predictions_final["t_prob"],
        1 / (1 - predictions_final["t_prob"]),
    )

    X = predictions_final[
        ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]
    ]

    abs_smd = calculate_covariate_balance(
        X, predictions_final["t"], predictions_final["weight"]
    )

    # track the average of weighted SMD on wandb
    avg_abs_smd_weighted = abs_smd.iloc[:, 0].mean()

    # track the Kolmogorov-Smirnov (KS) statistic for the propensity scores,
    # a lower KS statistic indicates better overlap between the groups.
    ks_statistic, _ = ks_2samp(
        predictions_final[predictions_final["t"] == 1]["t_prob"],
        predictions_final[predictions_final["t"] == 0]["t_prob"],
    )

    ATE_true = 1794.34
    ATE_IPTW = IPTW_unstabilized(
        predictions_final["t"], predictions_final["y"], predictions_final["t_prob"]
    )
    rmse_IPTW = rmse(ATE_IPTW, ATE_true)

    return {
        f"{prefix}: avg_abs_smd_weighted": avg_abs_smd_weighted,
        f"{prefix}: KS statistic": ks_statistic,
        f"{prefix}: RMSE from unstabilized IPTW": rmse_IPTW,
        f"{prefix}: average predicted t": predictions_final["t_prob"].mean(),
    }


def create_metric_plots(dataset: pd.DataFrame, dag: Dict, predictions: np.array, prefix: str, suffix: int) -> Dict[str, str]:
    # assign column names to the predictions_df
    causal_dataset = CausalDataset(dataset, dag)
    prediction_transformer = PredictionTransformer(causal_dataset.bin_edges)
    transformed_predictions = prediction_transformer.transform(predictions)

    predictions_final = pd.concat([dataset, transformed_predictions], axis=1)
    predictions_final["weight"] = np.where(
        predictions_final["t"] == 1,
        1 / predictions_final["t_prob"],
        1 / (1 - predictions_final["t_prob"]),
    )

    predictions_final["logodds_ps"] = np.log(predictions_final['t_prob'] / (1 - predictions_final['t_prob']))
    
    X = predictions_final[
        ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75", "u74", "u75"]
    ]
    abs_smd = calculate_covariate_balance(
        X, predictions_final["t"], predictions_final["weight"]
    )

    #plot absolute standardized mean difference
    fig, ax = plt.subplots(figsize=(8, 6))
    smd_plot(abs_smd, ax, epoch=suffix)

    smd_imagepath = 'experiments/results/figures/abs_smd.png'
    fig.savefig(smd_imagepath)

    # plot PS by treatment group
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = plot_propensity_score_distribution(
        predictions_final['t_prob'],
        predictions_final['t'],
        num_bins=100,
        reflect=True,
        kde=False,
        ax=ax,
        epoch=suffix
    )

    ps_imagepath = f'experiments/results/figures/propensity_score_distribution.png'
    fig.savefig(ps_imagepath)

    return {f"{prefix}: SMD_{suffix}": smd_imagepath,
            f"{prefix}: propensity score_{suffix}": ps_imagepath}


def images_to_gif(image_fnames: List[str], gif_outpath: str, duration: int = 5):
    image_fnames.sort(key=extract_number) #sort by step
    frames = [Image.open(image) for image in image_fnames]
    frame_one = frames[0]
    frame_one.save(gif_outpath, format="GIF", append_images=frames,
               save_all=True, duration=duration, loop=0)