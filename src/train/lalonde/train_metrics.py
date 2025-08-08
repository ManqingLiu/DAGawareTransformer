from typing import Dict
import numpy as np
import pandas as pd

from src.utils import rmse

# pseudo_ate is estimated using counterfactual cross-validation via causal forest (AIPW)
def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized mean-squared-error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2) / np.mean(y_true ** 2))

def std_nrmse(mu0: np.array, mu1: np.array, pseudo_ate: float) -> float:
    """Plug-in estimator, equivalent to standardization."""
    ite_pred = mu1 - mu0
    ate_pred = np.mean(ite_pred)
    return nrmse(pseudo_ate, ate_pred)

def ipw_nrmse(y: np.ndarray, t: np.ndarray, ps: np.ndarray, pseudo_ate: float) -> float:
    """Mean-squared-error with inverse propensity weighting"""
    ite_pred = (t * y / ps) - ((1 - t) * y / (1 - ps))
    ate_pred = np.mean(ite_pred)
    return nrmse(pseudo_ate, ate_pred)


def aipw_nrmse(y: np.ndarray, t: np.ndarray, mu0: np.array, mu1: np.array, ps: np.ndarray,  pseudo_ate: float) -> float:
    """Mean-squared-error with Counterfactual Cross Validation, equivalent to doubly robust estimator."""
    ite_pred = (t * (y - mu1) / ps) - (1 - t) * (y - mu0) / (1 - ps) + (mu1 - mu0)
    ate_pred = np.mean(ite_pred)
    return nrmse(pseudo_ate, ate_pred)

def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized mean-squared-error."""
    return np.mean((y_true - y_pred) ** 2) / np.mean(y_true ** 2)

def calculate_val_metrics(
    predictions: np.array,
    pseudo_ate: pd.DataFrame,
    sample_id: int,
    estimator: str,
    prefix: str
) -> Dict[str, float]:
    pseudo_ate_value = pseudo_ate.iloc[sample_id]["rmse_ate"]
    if estimator == "g-formula":
        std_nrmse_ = std_nrmse(predictions['pred_y_A0'], predictions['pred_y_A1'], pseudo_ate_value)
        return {
                f"{prefix}: NRMSE for standardization": std_nrmse_}
    elif estimator == "ipw":
        ipw_nrmse_ = ipw_nrmse(predictions['y'], predictions['t'], predictions['t_prob'], pseudo_ate_value)
        return {
                f"{prefix}: NRMSE for IPW": ipw_nrmse_}
    else:
        aipw_nrmse_ = aipw_nrmse(predictions['y'], predictions['t'], predictions['pred_y_A0'],
                                           predictions['pred_y_A1'], predictions['t_prob'], pseudo_ate_value)
        return {
                f"{prefix}: NRMSE for AIPW": aipw_nrmse_}