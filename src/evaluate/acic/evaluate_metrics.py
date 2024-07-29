from typing import Dict
import numpy as np
from src.utils import rmse

def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized mean-squared-error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2) / np.mean(y_true ** 2))

def std_nrmse(mu0: np.array, mu1: np.array, true_ite: np.array) -> float:
    """Plug-in estimator, equivalent to standardization."""
    ite_pred = mu1 - mu0
    ate_pred = np.mean(ite_pred)
    return ate_pred, nrmse(true_ite, ite_pred)


def ipw_nrmse(y: np.ndarray, t: np.ndarray, ps: np.ndarray, true_ite: np.array) -> float:
    """Mean-squared-error with inverse propensity weighting"""
    ite_pred = (t * y / ps) - ((1 - t) * y / (1 - ps))
    ate_pred = np.mean(ite_pred)
    return ate_pred, nrmse(true_ite, ite_pred)

def naive_ipw_nrmse(y: np.ndarray, t: np.ndarray, true_ite: np.array) -> float:
    """Mean-squared-error with inverse propensity weighting"""
    ps = np.mean(t)
    ite_pred = (t * y / ps) - ((1 - t) * y / (1 - ps))
    ate_pred = np.mean(ite_pred)
    return ate_pred, nrmse(true_ite, ite_pred)


def aipw_nrmse(y: np.ndarray, t: np.ndarray, mu0: np.array, mu1: np.array, ps: np.ndarray, true_ite: np.array) -> float:
    """Mean-squared-error with Counterfactual Cross Validation, equivalent to doubly robust estimator."""
    ite_pred = (t * (y - mu1) / ps) - (1 - t) * (y - mu0) / (1 - ps) + (mu1 - mu0)
    ate_pred = np.mean(ite_pred)
    return ate_pred, nrmse(true_ite, ite_pred)


def calculate_test_metrics_acic(
    predictions: np.array,
    ite: np.array,
    prefix: str,
    prop_score_threshold: float=0
) -> Dict[str, float]:
    if prop_score_threshold > 0:
        indices = predictions['t_prob'] < prop_score_threshold
        predictions.loc[indices, "t_prob"] = np.float32(prop_score_threshold)

        indices = predictions['t_prob'] > 1 - prop_score_threshold
        predictions.loc[indices, "t_prob"] = np.float32(1 - prop_score_threshold)

    ate_std, std_nrmse_ = std_nrmse(predictions['pred_y_A0'], predictions['pred_y_A1'], ite)
    ate_naive_ipw, naive_ipw_nrmse_ = naive_ipw_nrmse(predictions['y'], predictions['t'], ite)
    ate_ipw, ipw_nrmse_ = ipw_nrmse(predictions['y'], predictions['t'], predictions['t_prob'], ite)
    ate_aipw, aipw_nrmse_ = aipw_nrmse(predictions['y'], predictions['t'], predictions['pred_y_A0'],
                           predictions['pred_y_A0'], predictions['t_prob'], ite)

    return {
        f"{prefix}: predicted CATE for standardization: ": ate_std,
        f"{prefix}: NRMSE for standardization": std_nrmse_,
        f"{prefix}: predicted CATE for naive IPW": ate_naive_ipw,
        f"{prefix}: NRMSE for naive IPW": naive_ipw_nrmse_,
        f"{prefix}: predicted CATE for IPW": ate_ipw,
        f"{prefix}: NRMSE for IPW": ipw_nrmse_,
        f"{prefix}: predicted CATE for AIPW": ate_aipw,
        f"{prefix}: NRMSE for AIPW": aipw_nrmse_
    }