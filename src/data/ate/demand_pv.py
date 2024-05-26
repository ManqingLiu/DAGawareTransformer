import numpy as np
from numpy.random import default_rng

from src.data.ate.data_class import PVTrainDataSet, PVTestDataSet
import pandas as pd


def psi(t: np.ndarray) -> np.ndarray:
    return 2 * ((t - 5) ** 4 / 600 + np.exp(-4 * (t - 5) ** 2) + t / 10 - 2)


def generatate_demand_core(n_sample: int, rng, Z_noise: float = 1, W_noise: float = 1):
    demand = rng.uniform(0, 10, n_sample)
    cost1 = 2 * np.sin(demand * np.pi * 2 / 10) + rng.normal(0, Z_noise, n_sample)
    cost2 = 2 * np.cos(demand * np.pi * 2 / 10) + rng.normal(0, Z_noise, n_sample)
    price = 35 + (cost1 + 3) * psi(demand) + cost2 + rng.normal(0, 1.0, n_sample)
    views = 7 * psi(demand) + 45 + rng.normal(0, W_noise, n_sample)
    outcome = cal_outcome(price, views, demand)
    return demand, cost1, cost2, price, views, outcome


def generate_train_demand_pv(
    n_sample: int, Z_noise: float = 1, W_noise: float = 1, seed=42, **kwargs
):
    rng = default_rng(seed=seed)
    demand, cost1, cost2, price, views, outcome = generatate_demand_core(
        n_sample, rng, Z_noise, W_noise
    )
    outcome = (outcome + rng.normal(0, 1.0, n_sample)).astype(float)
    return PVTrainDataSet(
        treatment=price[:, np.newaxis],
        treatment_proxy=np.c_[cost1, cost2],
        outcome_proxy=views[:, np.newaxis],
        outcome=outcome[:, np.newaxis],
        backdoor=None,
    )


def cal_outcome(price, views, demand):
    return np.clip(np.exp((views - price) / 10.0), None, 5.0) * price - 5 * psi(demand)


def cal_structural(p: float, W_noise: float = 1):
    rng = default_rng(seed=42)
    demand = rng.uniform(0, 10.0, 10000)
    views = 7 * psi(demand) + 45 + rng.normal(0, W_noise, 10000)
    outcome = cal_outcome(p, views, demand)
    return np.mean(outcome)


def generate_test_demand_pv(W_noise: float = 1, **kwargs):
    price = np.linspace(10, 30, 10)
    treatment = np.array([cal_structural(p, W_noise) for p in price])
    return PVTestDataSet(
        structural=treatment[:, np.newaxis], treatment=price[:, np.newaxis]
    )


if __name__ == "__main__":
    train_data = generate_train_demand_pv(10000)
    average_outcome = np.mean(train_data.outcome)
    variance_outcome = np.var(train_data.outcome)
    print(f"{average_outcome:.2f}")

    # calculate average of 'outcome' for each 'treatment' from 10 to 30, binned by np.linspace(10, 30, 10)
    E_Y_givenA = np.array(
        [
            np.mean(
                train_data.outcome[
                    (train_data.treatment.flatten() >= p)
                    & (train_data.treatment.flatten() < p + (30 - 10) / 9)
                ]
            )
            for p in np.linspace(10, 30, 10)
        ]
    )

    true_EY_doA = np.array([cal_structural(a, 1) for a in np.linspace(10, 30, 10)])

    # Create a DataFrame with the desired columns
    df = pd.DataFrame({
        'A': np.linspace(10, 30, 10),
        'EY_givenA': E_Y_givenA,
        'EY_doA': true_EY_doA
    })

    # Save the DataFrame to a CSV file
    df.to_csv('experiments/results/proximal/E_Y_givenA.csv', index=False)
