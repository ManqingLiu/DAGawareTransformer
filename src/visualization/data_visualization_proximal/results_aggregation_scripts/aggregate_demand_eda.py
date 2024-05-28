import pandas as pd
import numpy as np
from numpy.random import default_rng

from src.data.ate.demand_pv import generatate_demand_core
from src.data.ate.demand_pv import cal_structural

"""
Generates 1000 random samples from the Demand experiment's data generating process.
Saves this data to a csv file in the results/ folder for visualization.
"""

n_samples = 1000
W_noise = 1
doA = np.linspace(10, 30, n_samples)
EY_doA = np.array([cal_structural(a, W_noise) for a in doA])
demand, cost1, cost2, price, views, obs_sales = generatate_demand_core(n_sample=n_samples, rng=default_rng(seed=42))

EDA_data = np.stack([demand, cost1, cost2, price, views, obs_sales, EY_doA, doA], axis=-1)
df = pd.DataFrame(EDA_data, columns=['U', 'Z1', 'Z2', 'A_obs', 'W', 'Y_obs', 'Y_struct', 'doA'])
df.to_csv("../../results/aggregated_results_for_figures/demand_eda_data.csv", index=False)
