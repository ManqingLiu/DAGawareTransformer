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
    ite_pred = (t * (y - mu1) / ps - (1 - t) * (y - mu0) / (1 - ps) + (mu1 - mu0))
    ate_pred = np.mean(ite_pred)
    return ate_pred, nrmse(true_ite, ite_pred)

def ipw_nrmse_stabilized(y: np.ndarray, t: np.ndarray, ps: np.ndarray, true_ite: np.ndarray) -> float:
    """Mean-squared-error with inverse propensity weighting and stabilized weights"""
    # Proportion of treatment and control groups
    prop_treatment = np.mean(t)
    prop_control = np.mean(1 - t)

    # Compute stabilized weights
    weights_treatment = prop_treatment / ps
    weights_control = prop_control / (1 - ps)

    # Calculate ITE predictions
    ite_pred = (t * y * weights_treatment) - ((1 - t) * y * weights_control)
    ate_pred = np.mean(ite_pred)

    return ate_pred, nrmse(true_ite, ite_pred)


def calculate_test_metrics_acic(
    predictions: np.array,
    ite: np.array,
    prefix: str,
    estimator: str,
    ps_lower_bound: float = 0.15,
    ps_upper_bound: float = 0.33
) -> Dict[str, float]:
    if estimator == "g-formula":
        ate_std, std_nrmse_ = std_nrmse(predictions['pred_y_A0'], predictions['pred_y_A1'], ite)
        return {f"{prefix}: predicted ATE for standardization": ate_std,
                f"{prefix}: NRMSE for standardization": std_nrmse_}
    elif estimator == "naive ipw":
        ate_naive_ipw, naive_ipw_nrmse_ = naive_ipw_nrmse(predictions['y'], predictions['t'], ite)
        return {f"{prefix}: predicted ATE for naive IPW": ate_naive_ipw,
                f"{prefix}: NRMSE for naive IPW": naive_ipw_nrmse_}
    elif estimator == "ipw":
        ate_ipw, ipw_nrmse_ = ipw_nrmse(predictions['y'], predictions['t'], predictions['t_prob'], ite)
        return {f"{prefix}: predicted ATE for IPW": ate_ipw,
                f"{prefix}: NRMSE for IPW": ipw_nrmse_}
    elif estimator == "ipw_stable":
        ate_ipw_stable, ipw_nrmse_stable_ = ipw_nrmse_stabilized(predictions['y'], predictions['t'], predictions['t_prob'], ite)
        return {f"{prefix}: predicted ATE for IPW with stabilized weights": ate_ipw_stable,
                f"{prefix}: NRMSE for IPW with stabilized weights": ipw_nrmse_stable_}
    else:
        # For t = 1, set t_prob to 0.33 if it's below 0.33
        predictions.loc[(predictions['t'] == 1) & (predictions['t_prob'] < ps_upper_bound), 't_prob'] = ps_upper_bound
        # For t = 0, set t_prob to 0.15 if it's below 0.15
        predictions.loc[(predictions['t'] == 0) & (predictions['t_prob'] < ps_lower_bound), 't_prob'] = ps_lower_bound
        ate_aipw, aipw_nrmse_ = aipw_nrmse(predictions['y'], predictions['t'], predictions['pred_y_A0'],
                                           predictions['pred_y_A1'], predictions['t_prob'], ite)
        return {f"{prefix}: predicted ATE for AIPW": ate_aipw,
                f"{prefix}: NRMSE for AIPW": aipw_nrmse_,
                f"{prefix}: mean of t_prob among treated": np.mean(predictions['t_prob'][predictions['t'] == 1]),
                f"{prefix}: mean of t_prob among control": np.mean(predictions['t_prob'][predictions['t'] == 0])}

