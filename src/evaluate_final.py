import pandas as pd
import json
import numpy as np
import os
from src.utils import rmse, log_results_evaluate
from argparse import ArgumentParser

def extract_results(estimator: str,
                    data_name: str,
                    mask: bool=True):
    # Initialize lists to hold the values
    values = []
    rnmse_values = []

    # Loop through the 50 sample files
    for i in range(1, 10):
        # Construct the file name based on the estimator
        if estimator == 'ipw' and mask == True:
            file_name = f'{data_name}_best_params_cfcv_sample{i}.json'
            predicted_ate_key = "Test: predicted CATE for IPW"
            nrmse_key = "Test: NRMSE for IPW"
        elif estimator == 'ipw' and mask == False:
            file_name = f'{data_name}_best_params_cfcv_nomask_sample{i}.json'
            predicted_ate_key = "Test: predicted CATE for IPW"
            nrmse_key = "Test: NRMSE for IPW"
        elif estimator == 'cfcv' and mask == True:
            file_name = f'{data_name}_best_params_cfcv_sample{i}.json'
            predicted_ate_key = "Test: predicted CATE for AIPW"
            nrmse_key = "Test: NRMSE for AIPW"
        elif estimator == 'cfcv' and mask == False:
            file_name = f'{data_name}_best_params_cfcv_nomask_sample{i}.json'
            predicted_ate_key = "Test: predicted CATE for AIPW"
            nrmse_key = "Test: NRMSE for AIPW"
        else:
            raise ValueError("Invalid estimator. Choose either 'ipw' or 'cfcv'.")

        # Load the JSON file
        with open(file_name, 'r') as file:
            data = json.load(file)

            # Extract the values and append to the lists
            value = data.get(predicted_ate_key)
            values.append(value)
            nrmse_value = data.get(nrmse_key)
            rnmse_values.append(nrmse_value)

    # Compute statistics
    mean_value = np.mean(values)
    quantile_25 = np.quantile(values, 0.25)
    quantile_75 = np.quantile(values, 0.75)
    standard_error_value = np.std(values) / np.sqrt(len(values))
    mean_rnmse = np.mean(rnmse_values)
    standard_error_rnmse = np.std(rnmse_values) / np.sqrt(len(rnmse_values))

    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        "Method": [estimator.upper()],
        "mean ATE": [mean_value],
        "se ATE": [standard_error_value],
        "CI lower": [quantile_25],
        "CI upper": [quantile_75],
        "mean RNMSE": [mean_rnmse],
        "se RNMSE": [standard_error_rnmse]
    })

    # Save the DataFrame to a CSV file
    results_df.to_csv(f'results_{data_name}_mask{mask}_{estimator}.csv', index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--estimator", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)

    args = parser.parse_args()

    # go to the directory where the results are stored (experiments/results/lalonde_psid)
    os.chdir(f'experiments/results/{args.data_name}')
    extract_results(args.estimator, args.data_name, mask=True)

