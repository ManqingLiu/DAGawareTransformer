import re
import os
import os.path as op
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.data.ate.demand_pv import cal_structural
from src.utils.misc_utils import sort_by_noise_level

"""
Aggregates the results from the Demand noise experiment.
 
* This script expects a folder named `data_for_demand_noiselevel_figures` within this directory (`data_visualization`)
* The folder should contain a sub-directory for each method. Each method folder should contain a unique sub-directory
    for each noise level in the experiment, for a total of 72 sub-directories/noise levels. 
* The name of these 72 directories should (automatically) contain a description of the noise levels.

Saves the aggregated results to csv files in the results/ folder
"""

cwd = os.getcwd()
data_for_figs = op.join(op.dirname(cwd), "data_for_demand_noiselevel_figures")
method_dirs = next(os.walk(data_for_figs))[1]

# Get true EY_doA
ticket_prices_coarse = np.linspace(10, 30, 10)
true_EY_doA = np.array([cal_structural(a, 1) for a in ticket_prices_coarse])

boxplot_df = pd.DataFrame(columns=['method', 'oos_mse', 'Z_noise', 'W_noise'])
predcurve_df = pd.DataFrame(columns=['rep', 'method', 'pred_EY_doA', 'true_EY_doA', 'A', 'Z_noise', 'W_noise'])
for method_dir in tqdm(method_dirs):

    method_name = method_dir.split()[0].lower()
    method_path = op.join(data_for_figs, method_dir)
    noise_dirs = next(os.walk(method_path))[1]
    noise_dirs = sort_by_noise_level(noise_dirs)

    for noise_dir in tqdm(noise_dirs, leave=False):

        noise_dir_path = op.join(data_for_figs, method_dir, noise_dir)
        noise_levels = re.findall(r'(\d+(?:\.\d+)?)', noise_dir)
        Z_noise = float(noise_levels[0])
        W_noise = float(noise_levels[1])

        boxplot_result_path = op.join(noise_dir_path, "result.csv")
        if op.isdir(noise_dir_path):
            oos_mse = np.loadtxt(boxplot_result_path)
            boxplot_df = boxplot_df.append(
                pd.DataFrame({'method': method_name, 'oos_mse': oos_mse, 'Z_noise': Z_noise, 'W_noise': W_noise}))

            for filename in os.listdir(noise_dir_path):
                if filename.endswith('.txt'):
                    pred_EY_doA = np.loadtxt(op.join(noise_dir_path, filename))
                    rep = int(re.match(r"(\d+).([a-z]+)", filename).groups()[0])
                    predcurve_df = predcurve_df.append(pd.DataFrame({'rep': rep,
                                                                     'method': method_name,
                                                                     'pred_EY_doA': pred_EY_doA,
                                                                     'true_EY_doA': true_EY_doA,
                                                                     'A': np.linspace(10, 30, 10),
                                                                     'Z_noise': Z_noise,
                                                                     'W_noise': W_noise}))
        else:
            raise NotImplementedError(f"{noise_dir_path=}")

boxplot_df.to_csv(op.join(data_for_figs, "../../results/aggregated_results_for_figures/demand_noise_boxplot_data.csv"), index=False)
predcurve_df.to_csv(op.join(data_for_figs, "../../results/aggregated_results_for_figures/demand_noise_predcurve_data.csv"), index=False)
