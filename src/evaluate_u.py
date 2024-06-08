import numpy as np
import pandas as pd
from src.utils import *
import json
from argparse import ArgumentParser









if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dag', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--predictions_file', type=str, required=True)
    #parser.add_argument('--results', type=str, required=True)

    args = parser.parse_args()

    with open(args.dag) as f:
        print(f'Loading dag file from {args.dag}')
        dag = json.load(f)

    with open(args.config) as f:
        print(f'Loading config file from {args.config}')
        config = json.load(f)

    data = pd.read_csv(args.data_file)
    predictions = pd.read_csv(args.predictions_file)
    print(predictions.describe())

    # print the percentage of A = 1
    print("Percentage of A = 1:", data['A'].mean())
    print("Percentage of A = 1:", predictions['A'].mean())
    # print the percentage of Y = 1
    print("Percentage of Y = 1:", data['Y'].mean())
    print("Percentage of Y = 1:", predictions['Y'].mean())

    # get true ATE: mean of y1 - mean of y0
    ATE_true = 0.182
    print("true ATE:", ATE_true)

    ATE_IPTW = IPTW_unstabilized(data['A'], data['Y'], predictions['A_prob'])
    print(f"Predicted ATE from IPTW: {ATE_IPTW:.4f}")
    rb_IPTW = relative_bias(ATE_IPTW, ATE_true)
    print(f"Relative bias from IPTW: {rb_IPTW:.4f}")