import numpy as np
import pandas as pd
from utils import *
import json
from argparse import ArgumentParser









if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dag', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--predictions_file', type=str, required=True)
    parser.add_argument('--results', type=str, required=True)

    args = parser.parse_args()

    with open(args.dag) as f:
        print(f'Loading dag file from {args.dag}')
        dag = json.load(f)

    with open(args.config) as f:
        print(f'Loading config file from {args.config}')
        config = json.load(f)

    data = pd.read_csv(args.data_file)
    predictions = pd.read_csv(args.predictions_file)
    predictions_final = pd.concat([data, predictions], axis=1)
    print(predictions_final.describe())




    # get true ATE: mean of y1 - mean of y0
    ATE_true = 0.462
    print("true ATE:", ATE_true)

    data = predictions_final.copy()

    pred_y_M1_A1 = data.loc[(data['M'] == 1) & (data['A'] == 1), 'pred_y']
    prob_A1 = np.mean(data['A'])
    m1_prob_A1 = data.loc[(data['M'] == 1) & (data['A'] == 1), 'm_prob']
    sum1 = np.mean(m1_prob_A1*pred_y_M1_A1*prob_A1)

    pred_y_M0_A1 = data.loc[(data['M'] == 0) & (data['A'] == 1), 'pred_y']
    m0_prob_A1 = data.loc[(data['M'] == 0) & (data['A'] == 1), 'm_prob']
    sum2 = np.mean(m0_prob_A1*pred_y_M0_A1*prob_A1)

    Y_A1 = sum1 + sum2

    pred_y_M1_A0 = data.loc[(data['M'] == 1) & (data['A'] == 0), 'pred_y']
    prob_A0 = 1 - prob_A1
    m1_prob_A0 = data.loc[(data['M'] == 1) & (data['A'] == 0), 'm_prob']
    sum1 = np.mean(m1_prob_A0*pred_y_M1_A0*prob_A0)

    pred_y_M0_A0 = data.loc[(data['M'] == 0) & (data['A'] == 0), 'pred_y']
    m0_prob_A0 = data.loc[(data['M'] == 0) & (data['A'] == 0), 'm_prob']
    sum2 = np.mean(m0_prob_A0*pred_y_M0_A0*prob_A0)

    Y_A0 = sum1 + sum2


    ATE_frontdoor = Y_A1 - Y_A0

    print(f"Predicted ATE from frontdoor: {ATE_frontdoor:.4f}")
    rmse_frontdoor = rmse(ATE_frontdoor, ATE_true)
    print(f"RMSE from frontdoor: {rmse_frontdoor:.4f}")




    # Gather results in a dictionary
    results = {
        'ATE_true': ATE_true,
        'ATE_frontdoor': ATE_frontdoor,
        'RMSE_frontdoor': rmse_frontdoor
    }


    # Log the results
    log_results(results, config, args.results)
