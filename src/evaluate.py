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
    ATE_true = data['y1'].mean() - data['y0'].mean()
    print("true ATE:", ATE_true)

    ATE_IPTW = IPTW_unstabilized(predictions_final['t'], predictions_final['y'], predictions_final['t_prob'])
    print(f"Predicted ATE from unstabilized IPTW: {ATE_IPTW:.4f}")
    #rb_IPTW = relative_bias(ATE_IPTW, ATE_true)
    #print(f"Relative bias from unstabilized IPTW: {rb_IPTW:.4f}")
    rmse_IPTW = rmse(ATE_IPTW, ATE_true)
    print(f"RMSE from unstabilized IPTW: {rmse_IPTW:.4f}")



    Y1_AIPW = AIPW_Y1(predictions_final['t'], predictions_final['t_prob'], predictions_final['y'], predictions_final['pred_y_A1'])
    Y0_AIPW = AIPW_Y0(predictions_final['t'], predictions_final['t_prob'], predictions_final['y'], predictions_final['pred_y_A0'])
    ATE_AIPW = Y1_AIPW - Y0_AIPW
    print(f"Estimated ATE from AIPW (DR): {ATE_AIPW:.4f}")
    #rb_AIPW = relative_bias(ATE_AIPW, ATE_true)
    #print(f"Relative bias from AIPW (DR): {rb_AIPW:.4f}")
    rmse_AIPW = rmse(ATE_AIPW, ATE_true)
    print(f"RMSE from AIPW (DR): {rmse_AIPW:.4f}")

    # Gather results in a dictionary
    results = {
        'ATE_true': ATE_true,
        'ATE_IPTW': ATE_IPTW,
        'RMSE_IPTW': rmse_IPTW,
        'ATE_AIPW': ATE_AIPW,
        'RMSE_AIPW': rmse_AIPW
    }


    # Log the results
    log_results_evaluate(results, config, args.results)