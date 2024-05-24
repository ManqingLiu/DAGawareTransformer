import re
import os
import os.path as op
import pandas as pd
import numpy as np
from src.data.ate.demand_pv import cal_structural

"""
Aggregates the results from the Demand experiment.

* This script expects a folder named `data_for_demand_figures` within this directory (`data_visualization`)
* The folder should contain a sub-directory for each method. Each method folder should contain a unique sub-directory
    for each sample size in the experiment: 1000, 5000, 10,000, and 50,000.
* The name of these 4 directories should (automatically) follow the convention: "n_sample:10000"

Saves the aggregated results to csv files in the results/ folder 
"""

# Get true EY_doA
ticket_prices_coarse = np.linspace(10, 30, 10)
true_EY_doA = np.array([cal_structural(a, 1) for a in ticket_prices_coarse])

cwd = os.getcwd()
data_for_figs = op.join(op.dirname(cwd), "data_for_demand_figures")
method_dirs = next(os.walk(data_for_figs))[1]

boxplot_df = pd.DataFrame(columns=['sample_size', 'method', 'oos_mse'])
predcurve_df = pd.DataFrame(columns=['sample_size', 'rep', 'method', 'pred_EY_doA', 'true_EY_doA', 'A'])
for method_dir in method_dirs:
    method_dirpath = op.join(data_for_figs, method_dir)

    result_dirs = ['n_sample:1000', 'n_sample:5000', 'n_sample:10000', 'n_sample:50000']

    for result_dir in result_dirs:

        method_name = method_dir.split()[0].lower()
        sample_size = int(result_dir.split(':')[1])
        result_dir_path = op.join(data_for_figs, method_dir, result_dir)

        boxplot_result_path = op.join(result_dir_path, "result.csv")
        if op.isdir(result_dir_path):
            oos_mse = np.loadtxt(boxplot_result_path)
            boxplot_df = boxplot_df.append(
                pd.DataFrame({'sample_size': sample_size, 'method': method_name, 'oos_mse': oos_mse}))

            for filename in os.listdir(result_dir_path):
                if filename.endswith('.txt'):
                    pred_EY_doA = np.loadtxt(op.join(result_dir_path, filename))
                    rep = int(re.match(r"(\d+).([a-z]+)", filename).groups()[0])
                    predcurve_df = predcurve_df.append(pd.DataFrame({'sample_size': sample_size, 'rep': rep,
                                                                     'method': method_name,
                                                                     'pred_EY_doA': pred_EY_doA,
                                                                     'true_EY_doA': true_EY_doA,
                                                                     'A': np.linspace(10, 30, 10)}))
        else:
            boxplot_df = boxplot_df.append(
                pd.DataFrame({'sample_size': [sample_size], 'method': [method_name], 'oos_mse': [999999]}))
            predcurve_df = predcurve_df.append(pd.DataFrame({'sample_size': sample_size, 'rep': 0,
                                                             'method': method_name,
                                                             'pred_EY_doA': 999999,
                                                             'true_EY_doA': true_EY_doA,
                                                             'A': np.linspace(10, 30, 10)}))
            continue

boxplot_df.to_csv(op.join(data_for_figs, "../../results/aggregated_results_for_figures/demand_boxplot_data.csv"), index=False)
predcurve_df.to_csv(op.join(data_for_figs, "../../results/aggregated_results_for_figures/demand_predcurve_data.csv"), index=False)
