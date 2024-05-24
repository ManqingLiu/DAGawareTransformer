import pandas as pd
import json
import os
from utils import rmse, log_results_evaluate
from argparse import ArgumentParser


def aggregate_data(data_name):
    # Initialize an empty list to hold data from all files
    data_list = []

    # Loop through each sample directory
    for i in range(100):
        directory = f'experiments/results/{data_name}/sample{i}'
        json_file_path = os.path.join(directory, f'{data_name}.json')
        nomask_json_file_path = os.path.join(directory, f'{data_name}_nomask.json')

        # Open and load the regular JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            ATE_AIPW_baseline = data[0]['results'].get('ATE_AIPW_baseline', None)
            results_dict = data[1]['results']
            ATE_true = results_dict.get('ATE_true', None)
            ATE_IPTW = results_dict.get('ATE_IPTW', None)
            ATE_AIPW = results_dict.get('ATE_AIPW', None)

        # Open and load the nomask JSON file

        with open(nomask_json_file_path, 'r') as file:
            nomask_data = json.load(file)
            nomask_results_dict = nomask_data[0]['results']
            ATE_IPTW_nomask = nomask_results_dict.get('ATE_IPTW', None)
            ATE_AIPW_nomask = nomask_results_dict.get('ATE_AIPW', None)

        # Append the extracted data to the list with new labels for nomask values
        data_list.append({
            'ATE_true': ATE_true,
            'ATE_AIPW_baseline': ATE_AIPW_baseline,
            'ATE_IPTW': ATE_IPTW,
            'ATE_AIPW': ATE_AIPW,
            'ATE_IPTW_nomask': ATE_IPTW_nomask,
            'ATE_AIPW_nomask': ATE_AIPW_nomask
        })

    # Create a DataFrame from the list
    df = pd.DataFrame(data_list)

    # Write the DataFrame to a CSV file
    csv_filename = f'experiments/results/{data_name}/{data_name}_results.csv'
    df.to_csv(csv_filename, index=False)
    print(f'Data aggregated and saved to {csv_filename}')

    return df


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_name', type=str, required=True)
    parser.add_argument('--results', type=str, required=True)

    args = parser.parse_args()
    df = aggregate_data(args.data_name)

    # calculate RMSE for ATE_AIPW_baseline using rmse function from utils.py
    # use mean of ATE_true and mean of ATE_AIPW_baseline as inputs
    ATE_true_mean = df['ATE_true'].mean()
    ATE_AIPW_baseline_mean = df['ATE_AIPW_baseline'].mean()
    rmse_AIPW_baseline = rmse(ATE_AIPW_baseline_mean, ATE_true_mean)
    print(f"RMSE for ATE_AIPW_baseline: {rmse_AIPW_baseline:.4f}")

    # calculate RMSE for ATE_IPTW using rmse function from utils.py
    # use mean of ATE_true and mean of ATE_IPTW as inputs
    ATE_IPTW_mean = df['ATE_IPTW'].mean()
    rmse_IPTW = rmse(ATE_IPTW_mean, ATE_true_mean)
    print(f"RMSE for ATE_IPTW: {rmse_IPTW:.4f}")

    # calculate RMSE for ATE_AIPW using rmse function from utils.py
    # use mean of ATE_true and mean of ATE_AIPW as inputs
    ATE_AIPW_mean = df['ATE_AIPW'].mean()
    rmse_AIPW = rmse(ATE_AIPW_mean, ATE_true_mean)
    print(f"RMSE for ATE_AIPW: {rmse_AIPW:.4f}")


    ATE_IPTW_nomask_mean = df['ATE_IPTW_nomask'].mean()
    rmse_IPTW_nomask = rmse(ATE_IPTW_nomask_mean, ATE_true_mean)
    ATE_AIPW_nomask_mean = df['ATE_AIPW_nomask'].mean()
    rmse_AIPW_nomask = rmse(ATE_AIPW_nomask_mean, ATE_true_mean)


    # Gather results in a dictionary
    results = {
        'ATE_true_mean': ATE_true_mean,
        'ATE_IPTW_mean': ATE_IPTW_mean,
        'ATE_AIPW_mean': ATE_AIPW_mean,
        'ATE_AIPW_baseline_mean': ATE_AIPW_baseline_mean,
        'ATE_IPTW_nomask_mean': ATE_IPTW_nomask_mean,
        'ATE_AIPW_nomask_mean': ATE_AIPW_nomask_mean,
        'RMSE_IPTW': rmse_IPTW,
        'RMSE_AIPW': rmse_AIPW,
        'RMSE_AIPW_baseline': rmse_AIPW_baseline,
        'RMSE_IPTW_nomask': rmse_IPTW_nomask,
        'RMSE_AIPW_nomask': rmse_AIPW_nomask
    }

    # Log the results
    log_results_evaluate(results, config=None, results_file=args.results)

