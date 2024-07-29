from typing import Dict
import numpy as np
from src.utils import rmse

def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized mean-squared-error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2) / np.mean(y_true ** 2))

def std_nrmse(mu0: np.array, mu1: np.array, true_ate=1794.34) -> float:
    """Plug-in estimator, equivalent to standardization."""
    ate_pred = np.mean(mu1 - mu0)
    return ate_pred, nrmse(true_ate, ate_pred)


def ipw_nrmse(y: np.ndarray, t: np.ndarray, ps: np.ndarray, true_ate=1794.34) -> float:
    """Mean-squared-error with inverse propensity weighting"""
    ate_pred = np.mean((t * y / ps) - ((1 - t) * y / (1 - ps)))
    return ate_pred, nrmse(true_ate, ate_pred)


def aipw_nrmse(y: np.ndarray, t: np.ndarray, mu0: np.array, mu1: np.array, ps: np.ndarray, true_ate=1794.34) -> float:
    """Mean-squared-error with Counterfactual Cross Validation, equivalent to doubly robust estimator."""
    ate_pred = np.mean((t * (y - mu1) / ps) - (1 - t) * (y - mu0) / (1 - ps) + (mu1 - mu0))
    return ate_pred, nrmse(true_ate, ate_pred)


def calculate_test_metrics(
    predictions: np.array,
    prefix: str,
    prop_score_threshold: float=0
) -> Dict[str, float]:
    if prop_score_threshold > 0:
        indices = predictions['t_prob'] < prop_score_threshold
        predictions.loc[indices, "t_prob"] = np.float32(prop_score_threshold)

        indices = predictions['t_prob'] > 1 - prop_score_threshold
        predictions.loc[indices, "t_prob"] = np.float32(1 - prop_score_threshold)

    ate_std, std_nrmse_ = std_nrmse(predictions['pred_y_A0'], predictions['pred_y_A1'])
    ate_ipw, ipw_nrmse_ = ipw_nrmse(predictions['y'], predictions['t'], predictions['t_prob'])
    ate_aipw, aipw_nrmse_ = aipw_nrmse(predictions['y'], predictions['t'], predictions['pred_y_A0'],
                           predictions['pred_y_A0'], predictions['t_prob'])

    return {
        f"{prefix}: predicted ATE for standardization: ": ate_std,
        f"{prefix}: NRMSE for standardization": std_nrmse_,
        f"{prefix}: predicted ATE for IPW": ate_ipw,
        f"{prefix}: NRMSE for IPW": ipw_nrmse_,
        f"{prefix}: predicted ATE for AIPW": ate_aipw,
        f"{prefix}: NRMSE for AIPW": aipw_nrmse_
    }