import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def process_data(data_name):
    # Load datasets
    baseline = pd.read_csv(f'experiments/results/{data_name}/grf_baseline_results.csv')
    ipw = pd.read_csv(f'experiments/results/{data_name}/results_{data_name}_maskTrue_ipw.csv')
    ipw_nomask = pd.read_csv(f'experiments/results/{data_name}/results_{data_name}_maskFalse_ipw.csv')
    cfcv = pd.read_csv(f'experiments/results/{data_name}/results_{data_name}_maskTrue_cfcv.csv')
    cfcv_nomask = pd.read_csv(f'experiments/results/{data_name}/results_{data_name}_maskFalse_cfcv.csv')

    # Standardize column names
    columns_standard = ['Method', 'Mean_ATE', 'se_ATE', 'CI_Lower', 'CI_Upper', 'Mean_RNMSE', 'se_RNMSE']
    ipw_nomask.columns = columns_standard
    ipw_nomask['Method'] = 'IPW (VT)'
    ipw.columns = columns_standard
    ipw['Method'] = 'IPW (Ours)'
    cfcv_nomask.columns = columns_standard
    cfcv_nomask['Method'] = 'AIPW (VT)'
    cfcv.columns = columns_standard
    cfcv['Method'] = 'AIPW (Ours)'

    # Combine datasets
    combined_df = pd.concat([baseline, ipw_nomask, ipw, cfcv_nomask, cfcv])

    # Define the desired order for the 'Method' column
    method_order = ['IPW (Naive)', 'IPW (GRF)', 'IPW (VT)', 'IPW (Ours)', 'AIPW (GRF)', 'AIPW (VT)', 'AIPW (Ours)']

    # Convert the 'Method' column to a categorical type with the specified order
    combined_df['Method'] = pd.Categorical(combined_df['Method'], categories=method_order, ordered=True)

    # Sort the DataFrame by the 'Method' column
    combined_df = combined_df.sort_values('Method')

    # save results as a csv
    combined_df.to_csv(f'experiments/results/{data_name}/{data_name}_combined_results.csv', index=False)

    # Plot Mean RNMSE with error bars
    plt.figure(figsize=(10, 5))
    plt.bar(combined_df["Method"], combined_df["Mean_RNMSE"], yerr=combined_df["se_RNMSE"], capsize=5)
    plt.xlabel('Method')
    plt.ylabel('Mean RNMSE')
    plt.title('Mean RNMSE with Standard Error')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save plot as png in experiments/results/data_name
    plt.savefig(f'experiments/results/{data_name}/{data_name}_mean_rnmse.png')
    plt.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_name', type=str, required=True)
    args = parser.parse_args()

    process_data(args.data_name)