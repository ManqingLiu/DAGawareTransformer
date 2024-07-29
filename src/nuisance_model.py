from argparse import ArgumentParser
import json
import numpy as np

from econml.dr import DRLearner
from src.data.data_preprocess import data_preprocess



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    filepaths = config["filepaths"]
    config_train = config["training"]

    with open(filepaths["dag"]) as f:
        dag = json.load(f)

    (train_data, train_dataloader, val_data, val_dataloader, test_data, test_dataloader) = data_preprocess(
        config, filepaths, dag
    )

    np.random.seed(42)
    # X is the covariates in val_data
    X = val_data[["age","education","black","hispanic","married","nodegree","re74","re75","u74","u75"]].to_numpy()
    T = val_data[["t"]].to_numpy()
    y = val_data[["y"]].to_numpy().ravel()
    est = DRLearner()
    est.fit(y, T, X=X, W=None)