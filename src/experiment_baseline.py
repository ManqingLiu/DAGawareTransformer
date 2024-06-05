from argparse import ArgumentParser

import pandas as pd

from causallib.datasets import load_nhefs
from causallib.estimation import IPW
from causallib.evaluation import evaluate

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from matplotlib import pyplot as plt

from src.dataset import DataObject
from utils import rmse

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_train_file', type=str, required=True)
    parser.add_argument('--data_val_file', type=str, required=True)
    parser.add_argument('--data_test_file', type=str, required=True)
    args = parser.parse_args()

    data_train = pd.read_csv(args.data_train_file)
    data_val = pd.read_csv(args.data_val_file)
    data_test = pd.read_csv(args.data_test_file)

    # print column names of data_train
    print(data_train.columns)

    # Select the relevant columns for X, a, and y
    X = data_train[['age', 'education', 'black', 'hispanic', 'married', 'nodegree', 're74', 're75']]
    a = data_train['t']
    y = data_train['y']

    # Create an instance of DataObject
    data = DataObject(X, a, y)

    learner = LogisticRegression(solver="liblinear")

    ipw = IPW(learner)

    ipw.fit(X, a)
    outcomes = ipw.estimate_population_outcome(X, a, y)
    effect = ipw.estimate_effect(outcomes[1], outcomes[0], effect_types=["diff"])
    print(effect)

    # evaluate on test data
    ipw = IPW(learner)
    X_test = data_test[['age', 'education', 'black', 'hispanic', 'married', 'nodegree', 're74', 're75']]
    a_test = data_test['t']
    y_test = data_test['y']
    plots = ["covariate_balance_love", "weight_distribution"]
    evaluation_results = evaluate(ipw, X_test, a_test, y_test)
    outcomes_test = ipw.estimate_population_outcome(X_test, a_test, y_test)
    effect_test = ipw.estimate_effect(outcomes_test[1], outcomes_test[0], effect_types=["diff"])
    print(effect_test)
    f, [a0, a1] = plt.subplots(2, 1, figsize=(10, 12))
    evaluation_results.plot_covariate_balance(kind="love", ax=a0)
    evaluation_results.plot_weight_distribution(ax=a1)
    plt.suptitle("Evaluation on test data")
    # show plot
    plt.show()

    # save plot as png
    f.savefig('evaluation_test.png')

    ATE_true = data_test['y1'].mean() - data_test['y0'].mean()
    print(f"True ATE: {ATE_true}")
    rmse_test = rmse(effect_test, ATE_true)
    print(f"RMSE on test data: {rmse_test}")








