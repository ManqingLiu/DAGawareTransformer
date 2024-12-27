import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser

def draw_histograms(data_name):
    estimators = ['G-formula', 'IPW', 'AIPW']

    methods = {
        'G-formula': ['GRF', 'MLP', 'Ours'],
        'IPW': ['GRF', 'MLP', 'Ours'],
        'AIPW': ['GRF', 'MLP', 'Ours \n (Sep)', 'Ours \n (Joint)']
    }

    if data_name == "lalonde_cps":

        means = {
            'G-formula': [0.925, 0.922, 0.901],
            'IPW': [6.342, 0.631, 0.614],
            'AIPW': [1.596, 0.517, 1.512, 0.351]
        }
    
        errors = {
            'G-formula': [0.00258, 0.00632, 0.00534],
            'IPW': [1.227, 0.104, 0.100],
            'AIPW': [0.294, 0.142, 0.578, 0.130]
        }
    elif data_name == "lalonde_psid":
        means = {
            'G-formula': [1.009, 0.979, 0.964],
            'IPW': [9.408, 2.258, 2.198],
            'AIPW': [2.517, 1.249, 3.237, 0.922]
        }

        errors = {
            'G-formula': [0.021, 0.004, 0.014],
            'IPW': [1.108, 0.745, 0.705],
            'AIPW': [0.242, 0.259, 0.571, 0.264]
        }
    else:
        means = {
            'G-formula': [0.346, 0.417, 0.344],
            'IPW': [4.281, 4.142, 4.102],
            'AIPW': [0.857, 1.304, 0.698, 0.814]
        }

        errors = {
            'G-formula': [0.044, 0.062, 0.054],
            'IPW': [1.388, 1.278, 1.272],
            'AIPW': [0.059, 0.112, 0.057, 0.052]
        }

    # Colors for each method
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Create the plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    #fig.suptitle('Lalonde-cps Results by Estimator', fontsize=20)

    for i, estimator in enumerate(estimators):
        ax = axes[i]
        x = np.arange(len(methods[estimator]))

        ax.bar(x, means[estimator], yerr=errors[estimator], align='center',
               alpha=0.8, ecolor='black', capsize=5, color=colors[:len(methods[estimator])])

        ax.set_ylabel('Mean NRMSE')
        ax.set_title(estimator)
        ax.set_xticks(x)
        ax.set_xticklabels(methods[estimator], rotation=0, ha='right', fontsize=15)

        # Add value labels on top of each bar
        for j, v in enumerate(means[estimator]):
            ax.text(j, v + errors[estimator][j], f'{v:.3f}', ha='center', va='bottom')

        ax.set_ylim(0, max(max(means[estimator]) + max(errors[estimator]), 7) * 1.1)  # Adjust y-axis limit

    # Add a common legend
    fig.legend(methods['AIPW'], loc='upper center', bbox_to_anchor=(0.5, 0), ncol=4)

    plt.tight_layout()
    plt.savefig(f'experiments/results/{data_name}/{data_name}.png')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True
    )
    args = parser.parse_args()
    draw_histograms(args.data_name)