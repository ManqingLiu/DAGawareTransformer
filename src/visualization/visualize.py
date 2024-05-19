
# Realcause baseline
# PSID
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser


def ate_plot(dataset):
    # Define file paths
    results_path = f'experiments/results/{dataset}/{dataset}_results.csv'
    baseline_path = 'realcause_baseline.csv'

    # Load data
    dataset_results = pd.read_csv(results_path)
    realcause_baseline = pd.read_csv(baseline_path)

    # Calculate ate_estimate
    ate_mean = dataset_results['ATE_true'].mean()

    def calculate_ate_estimate(row):
        if row['ate_bias'] >= 0:
            return ate_mean + row['ate_rmse']
        else:
            return ate_mean - row['ate_rmse']

    realcause_baseline['ate_estimate'] = realcause_baseline.apply(calculate_ate_estimate, axis=1)

    # Calculate 95% CI for ate_estimate
    realcause_baseline['ate_estimate_lower'] = realcause_baseline['ate_estimate'] - 1.96 * realcause_baseline[
        'ate_std_error']
    realcause_baseline['ate_estimate_upper'] = realcause_baseline['ate_estimate'] + 1.96 * realcause_baseline[
        'ate_std_error']

    # Filter data for the specific dataset
    realcause_baseline = realcause_baseline[realcause_baseline['dataset'] == dataset]

    # Keep specific columns and set the first column as index
    realcause_baseline = realcause_baseline[
        ['prop_score_model', 'ate_estimate', 'ate_std_error', 'ate_estimate_lower', 'ate_estimate_upper']]
    realcause_baseline.set_index('prop_score_model', inplace=True)
    realcause_baseline.index.name = None

    # Calculate mean and std of each column in dataset_results
    model_results = dataset_results.describe().loc[['mean', 'std']].T
    model_results['lower'] = model_results['mean'] - 1.96 * model_results['std']
    model_results['upper'] = model_results['mean'] + 1.96 * model_results['std']

    # Rename columns in realcause_baseline to match those in model_results
    realcause_baseline.columns = ['mean', 'std', 'lower', 'upper']

    # Append realcause_baseline to model_results
    model_results = model_results._append(realcause_baseline, ignore_index=False)

    # Reset the index
    model_results.reset_index(inplace=True)

    # Rename the column
    model_results.rename(columns={'index': 'estimator'}, inplace=True)

    # Save the results to a CSV file
    model_results.to_csv(f'experiments/results/{dataset}/model_results_plot.csv', index=False)

    # Prepare data for plotting
    # Define a dictionary to map dataset names to order and labels
    dataset_dict = {
        'lalonde_cps': {
            'order': ["LogisticRegression", "DecisionTree", "ATE_IPTW_nomask", "ATE_IPTW", "ATE_AIPW_baseline", "ATE_AIPW_nomask",
                      "ATE_AIPW"],
            'labels': ["IPTW (LR)", "IPTW (Decision Tree)", "IPTW (VT)", "IPTW (Ours)", "AIPW (Baseline)", "AIPW (VT)",
                       "AIPW (Ours)"]
        },
        'lalonde_psid': {
            'order': ["LogisticRegression", "kNN", "ATE_IPTW_nomask", "ATE_IPTW", "ATE_AIPW_baseline", "ATE_AIPW_nomask", "ATE_AIPW"],
            'labels': ["IPTW (LR)", "IPTW (KNN)", "IPTW (VT)", "IPTW (Ours)", "AIPW (Baseline)", "AIPW (VT)", "AIPW (Ours)"]
        },
        'twins': {
            'order': ["LogisticRegression", "LogisticRegression_l2_liblinear", "ATE_IPTW_nomask", "ATE_IPTW", "ATE_AIPW_baseline",
                      "ATE_AIPW_nomask", "ATE_AIPW"],
            'labels': ["IPTW (LR)", "IPTW (Ridge Regression)", "IPTW (VT)", "IPTW (Ours)", "AIPW (Baseline)", "AIPW (VT)",
                       "AIPW (Ours)"]
        }
    }

    # Use the dataset name to look up the correct order and labels
    order = dataset_dict[dataset]['order']
    labels = dataset_dict[dataset]['labels']
    plot_data = model_results[model_results['estimator'].isin(order)]
    plot_data = plot_data.set_index("estimator").loc[order].reset_index()

    # Plotting
    means = plot_data['mean']
    ci_lowers = means - plot_data['lower']
    ci_uppers = plot_data['upper'] - means
    fig, ax = plt.subplots()
    ax.errorbar(labels, means, yerr=[ci_lowers, ci_uppers], fmt='o', capsize=5, color='blue', ecolor='lightblue')
    ax.set_ylabel('ATE')
    ax.axhline(y=model_results[model_results['estimator'] == "ATE_true"]['mean'].values[0], color='r',
               linestyle='dotted', label='True ATE')
    plt.xticks(rotation=45)
    plt.title('Comparison of ATE Estimates with 95% CI')
    plt.legend()
    plt.tight_layout()

    # Save plot to file
    plot_path = f'experiments/results/{dataset}/model_results_plot.png'
    plt.savefig(plot_path)
    plt.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_name', type=str, required=True)

    args = parser.parse_args()
    ate_plot(args.data_name)



