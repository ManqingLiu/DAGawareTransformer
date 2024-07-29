from typing import Dict
import numpy as np

from src.utils import rmse

# pseudo_ate is estimated using counterfactual cross-validation via causal forest (AIPW)
def std_rmse(mu0: np.array, mu1: np.array, pseudo_ate: float) -> float:
    """Plug-in estimator, equivalent to standardization."""
    ite_pred = mu1 - mu0
    ate_pred = np.mean(ite_pred)
    return rmse(pseudo_ate, ate_pred)

def ipw_rmse(y: np.ndarray, t: np.ndarray, ps: np.ndarray, pseudo_ate: float) -> float:
    """Mean-squared-error with inverse propensity weighting"""
    ite_pred = (t * y / ps) - ((1 - t) * y / (1 - ps))
    ate_pred = np.mean(ite_pred)
    return rmse(pseudo_ate, ate_pred)


def cfcv_rmse(y: np.ndarray, t: np.ndarray, mu0: np.array, mu1: np.array, ps: np.ndarray,  pseudo_ate: float) -> float:
    """Mean-squared-error with Counterfactual Cross Validation, equivalent to doubly robust estimator."""
    ite_pred = (t * (y - mu1) / ps) - (1 - t) * (y - mu0) / (1 - ps) + (mu1 - mu0)
    ate_pred = np.mean(ite_pred)
    return rmse(pseudo_ate, ate_pred)

def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized mean-squared-error."""
    return np.mean((y_true - y_pred) ** 2) / np.mean(y_true ** 2)

def calculate_val_metrics(
    predictions: np.array,
    prefix: str,
    prop_score_threshold: float=0
) -> Dict[str, float]:
    if prop_score_threshold > 0:
        indices = predictions['t_prob'] < prop_score_threshold
        predictions.loc[indices, "t_prob"] = prop_score_threshold

        indices = predictions['t_prob'] > 1 - prop_score_threshold
        predictions.loc[indices, "t_prob"] = 1 - prop_score_threshold

    ps_trt = np.mean(predictions['t_prob'][predictions['t'] == 1])
    ps_ctrl = np.mean(predictions['t_prob'][predictions['t'] == 0])
    std_rmse_ = std_rmse(predictions['pred_y_A0'], predictions['pred_y_A1'], predictions['t'])
    ipw_rmse_ = ipw_rmse(predictions['y'], predictions['t'], predictions['t_prob'])
    cfcv_rmse_ = cfcv_rmse(predictions['y'], predictions['t'], predictions['pred_y_A0'],
                           predictions['pred_y_A0'], predictions['t_prob'])

    return {
        f"{prefix}: Propensity score for treatment (mean)": ps_trt,
        f"{prefix}: Propensity score for control (mean)": ps_ctrl,
        f"{prefix}: RMSE for standardization": std_rmse_,
        f"{prefix}: RMSE for IPW": ipw_rmse_,
        f"{prefix}: RMSE for CFCV": cfcv_rmse_
    }