from typing import Dict
import numpy as np
from sklearn.metrics import mean_squared_error as mse

from src.utils import rmse

def std_rmse(mu0: np.array, mu1: np.array) -> float:
    """Plug-in estimator, equivalent to standardization."""
    ate_pred = np.mean(mu1 - mu0)
    pseudo_ate = 766.29
    return rmse(pseudo_ate, ate_pred)


def ipw_rmse(y: np.ndarray, t: np.ndarray, ps: np.ndarray) -> float:
    """Mean-squared-error with inverse propensity weighting"""
    ate_pred = np.mean((t * y / ps) - ((1 - t) * y / (1 - ps)))
    pseudo_ate = 739.34
    return rmse(pseudo_ate, ate_pred)


def cfcv_rmse(y: np.ndarray, t: np.ndarray, mu0: np.array, mu1: np.array, ps: np.ndarray) -> float:
    """Mean-squared-error with Counterfactual Cross Validation, equivalent to doubly robust estimator."""
    pseudo_ate = 1422.82
    ate_pred = np.mean((t * (y - mu1) / ps) - ((1 - t) * (y - mu0) / (1 - ps)) + (mu1 - mu0))
    return rmse(pseudo_ate, ate_pred)

def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized mean-squared-error."""
    return np.mean((y_true - y_pred) ** 2) / np.mean(y_true ** 2)

def calculate_metrics(
    predictions: np.array, prefix: str
) -> Dict[str, float]:
    std_rmse_ = std_rmse(predictions['pred_y_A0'], predictions['pred_y_A1'])
    ipw_rmse_ = ipw_rmse(predictions['y'], predictions['t'], predictions['t_prob'])
    cfcv_rmse_ = cfcv_rmse(predictions['y'], predictions['t'], predictions['pred_y_A0'],
                        predictions['pred_y_A1'], predictions['t_prob'])

    return {
        f"{prefix}: RMSE for standardization": std_rmse_,
        f"{prefix}: RMSE for IPW": ipw_rmse_,
        f"{prefix}: RMSE for CFCV": cfcv_rmse_
    }