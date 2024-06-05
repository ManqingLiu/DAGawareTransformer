import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    predictions_final['weight'] = np.where(predictions_final['t'] == 1,
                                           1 / predictions_final['t_prob'],
                                           1 / (1 - predictions_final['t_prob']))

    X = predictions_final[['age', 'education', 'black', 'hispanic', 'married', 'nodegree', 're74', 're75']]

    abs_smd = calculate_covariate_balance(X,
                                          predictions_final['t'],
                                          predictions_final['weight'])

    print(abs_smd)

    # plot absolute standardized mean difference
    fig, ax = plt.subplots(figsize=(8, 6))
    smd_plot(abs_smd, ax)

    plt.show()
    # save the plot as png
    fig.savefig('experiments/results/figures/abs_smd.png')

    # plot PS by treatment group
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_propensity_score_distribution(
        predictions_final['t_prob'],
        predictions_final['t'],
        reflect=True,
        kde=False
    )

    plt.show()
    fig.savefig('experiments/results/figures/propensity_score_distribution.png')


    # get true ATE: mean of y1 - mean of y0
    ATE_true = data['y1'].mean() - data['y0'].mean()
    print("true ATE:", ATE_true)

    # calculate naive ATE which is the difference of mean of y for t=1 and t=0
    ATE_naive = predictions_final[predictions_final['t'] == 1]['y'].mean() - predictions_final[predictions_final['t'] == 0]['y'].mean()
    print("naive ATE:", ATE_naive)
    rmse_naive = rmse(ATE_naive, ATE_true)
    print(f"RMSE from naive ATE: {rmse_naive:.4f}")

    # calcualte t_prob marginal which is the mean of t (from 0 to 1) of t in predictions_final
    t_prob_marginal = predictions_final['t'].mean()
    # print average of t
    print(f"Average of t: {t_prob_marginal:.4f}")
    # assign t_prob_marginal to predictions_final
    predictions_final['t_prob_marginal'] = t_prob_marginal
    # calculate ATE IPTW using the marginal t_prob
    ATE_IPTW_marginal = IPTW_unstabilized(predictions_final['t'], predictions_final['y'], predictions_final['t_prob_marginal'])
    print(f"Predicted ATE from IPTW with marginal t_prob: {ATE_IPTW_marginal:.4f}")
    rmse_IPTW_marginal = rmse(ATE_IPTW_marginal, ATE_true)
    print(f"RMSE from IPTW with marginal t_prob: {rmse_IPTW_marginal:.4f}")

    ATE_IPTW = IPTW_unstabilized(predictions_final['t'], predictions_final['y'], predictions_final['t_prob'])
    print(f"Predicted ATE from unstabilized IPTW: {ATE_IPTW:.4f}")
    rmse_IPTW = rmse(ATE_IPTW, ATE_true)
    print(f"RMSE from unstabilized IPTW: {rmse_IPTW:.4f}")

    ATE_IPTW_stab = IPTW_stabilized(predictions_final['t'], predictions_final['y'], predictions_final['t_prob'])
    print(f"Predicted ATE from stabilized IPTW: {ATE_IPTW_stab:.4f}")
    rmse_IPTW_stab = rmse(ATE_IPTW_stab, ATE_true)
    print(f"RMSE from stabilized IPTW: {rmse_IPTW_stab:.4f}")

    ATE_std = standardization(predictions_final['pred_y_A1'], predictions_final['pred_y_A0'])
    print(f"Estimated ATE from standardization: {ATE_std:.4f}")
    rmse_std = rmse(ATE_std, ATE_true)
    print(f"RMSE from standardization: {rmse_std:.4f}")

    Y1_AIPW = AIPW_Y1(predictions_final['t'], predictions_final['t_prob'], predictions_final['y'], predictions_final['pred_y_A1'])
    Y0_AIPW = AIPW_Y0(predictions_final['t'], predictions_final['t_prob'], predictions_final['y'], predictions_final['pred_y_A0'])
    ATE_AIPW = Y1_AIPW - Y0_AIPW
    print(f"Estimated ATE from AIPW (DR): {ATE_AIPW:.4f}")
    rmse_AIPW = rmse(ATE_AIPW, ATE_true)
    print(f"RMSE from AIPW (DR): {rmse_AIPW:.4f}")

    # Gather results in a dictionary
    results = {
        'ATE_true': ATE_true,
        'ATE_IPTW': ATE_IPTW,
        'RMSE_IPTW': rmse_IPTW,
        'ATE_std': ATE_std,
        'RMSE_std': rmse_std,
        'ATE_AIPW': ATE_AIPW,
        'RMSE_AIPW': rmse_AIPW
    }




    # Log the results
    log_results_evaluate(results, config, args.results)
