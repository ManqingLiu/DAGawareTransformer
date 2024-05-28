import os
import os.path as op
import re
import pandas as pd
import numpy as np

from src.data.ate.dsprite import generate_test_dsprite

"""
Aggregates the results from the dSprite experiment.

* This script expects a folder named `data_for_dsprite_figures` within this directory (`data_visualization`)
* The folder should contain a sub-directory for each method. Each method folder should contain a unique sub-directory
    for each sample in the experiment: 1000, 5000, and 7500. 
* The name of these directories should follow the convention: "n_sample:7500"

Saves the aggregated results to a csv file in the results/ folder 
"""

dsprite_test_data = generate_test_dsprite()
true_EY_doA = dsprite_test_data.structural.squeeze()

cwd = os.getcwd()
data_for_figs = op.join(op.dirname(cwd), "data_for_dsprite_figures")
method_dirs = next(os.walk(data_for_figs))[1]

boxplot_df = pd.DataFrame(columns=['sample_size', 'method', 'oos_mse'])
predcurve_df = pd.DataFrame(columns=['sample_size', 'rep', 'method', 'pred_EY_doA', 'true_EY_doA', 'A'])
for method_dir in method_dirs:
    method_dirpath = op.join(data_for_figs, method_dir)

    result_dirs = ['n_sample:1000', 'n_sample:5000', 'n_sample:7500']
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
                                                                     'A': np.linspace(1, 588, 588)}))

        else:
            boxplot_df = boxplot_df.append(
                pd.DataFrame({'sample_size': [sample_size], 'method': [method_name], 'oos_mse': [999999]}))
            predcurve_df = predcurve_df.append(pd.DataFrame({'sample_size': sample_size, 'rep': 0,
                                                             'method': method_name,
                                                             'pred_EY_doA': 999999,
                                                             'true_EY_doA': true_EY_doA,
                                                             'A': np.linspace(1, 588, 588)}))
            continue

boxplot_df.to_csv(op.join(data_for_figs, "../../results/aggregated_results_for_figures/dsprite_boxplot_data.csv"), index=False)
predcurve_df.to_csv(op.join(data_for_figs, "../../results/aggregated_results_for_figures/dsprite_predcurve_data.csv"), index=False)
