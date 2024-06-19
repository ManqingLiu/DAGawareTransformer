import os
import json
from argparse import ArgumentParser

import torch
import ray
from ray import train, tune
from ray.tune import Stopper

from experiments.tuning.config import config_psid
from src.utils import log_results
from src.data.data_preprocess import data_preprocess
from src.experiment import experiment
from src.train.lalonde_psid.train_metrics import std_rmse, ipw_rmse, cfcv_rmse

class EarlyStoppingAtMinRMSE(Stopper):
    def __init__(self, patience=5):
        self.best_rmse = float("inf")
        self.no_improvement_count = 0
        self.patience = patience

    def __call__(self, trial_id, result):
        current_rmse = result["cfcv_rmse"]
        if current_rmse < self.best_rmse:
            self.best_rmse = current_rmse
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        return self.no_improvement_count >= self.patience

    def stop_all(self):
        return False


def fine_tune(config, dag, train_dataloader, val_dataloader, val_data):
    model_trained, train_loss, val_loss, predictions = experiment(config, dag,
                                                                  train_dataloader,
                                                                  val_dataloader, val_data, random_seed=False)
    std_rmse_ = std_rmse(predictions['pred_y_A0'], predictions['pred_y_A1'])
    ipw_rmse_ = ipw_rmse(predictions['y'], predictions['t'], predictions['t_prob'])
    cfcv_rmse_ = cfcv_rmse(predictions['y'], predictions['t'], predictions['pred_y_A0'],
                        predictions['pred_y_A1'], predictions['t_prob'])

    train.report({"train_loss": train_loss, "val_loss": val_loss, "std_rmse":  std_rmse_,
                  "ipw_rmse": ipw_rmse_, "cfcv_rmse": cfcv_rmse_})



if __name__ == '__main__':
    current_directory = os.getcwd()
    print(f"Current directory: {current_directory}")
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, default="config/train/lalonde_psid.json"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    filepaths = config["filepaths"]

    with open(filepaths["dag"]) as f:
        dag = json.load(f)

    (train_data, train_dataloader, val_data, val_dataloader, test_data, test_dataloader) = data_preprocess(
        config, filepaths, dag
    )

    fine_tune_new = tune.with_parameters(fine_tune,
                                         dag=dag,
                                         train_dataloader=train_dataloader,
                                         val_dataloader=val_dataloader,
                                         val_data=val_data)

    analysis = tune.run(
        fine_tune_new,
        config=config_psid,
        num_samples=30,  # Number of times to sample from the hyperparameter space
        resources_per_trial={"cpu": 1, "gpu": 1 if torch.cuda.is_available() else 0},
        stop={"cfcv_rmse": 1000}
    )

    best_trial_std = analysis.get_best_trial("std_rmse", "min", "last")
    print("Best trial config: {}".format(best_trial_std.config))
    print("Best trial final validation std RMSE: {}".format(best_trial_std.last_result["std_rmse"]))
    best_trial_ipw = analysis.get_best_trial("ipw_rmse", "min", "last")
    print("Best trial config: {}".format(best_trial_ipw.config))
    print("Best trial final validation IPW RMSE: {}".format(best_trial_ipw.last_result["ipw_rmse"]))
    best_trial_cfcv = analysis.get_best_trial("cfcv_rmse", "min", "last")
    print("Best trial config: {}".format(best_trial_cfcv.config))
    print("Best trial final validation CFCV RMSE: {}".format(best_trial_cfcv.last_result["cfcv_rmse"]))
    # Log the results
    log_results(best_trial_std.config, filepaths["result_file"])
    log_results(best_trial_ipw.config, filepaths["result_file"])
    log_results(best_trial_cfcv.config, filepaths["result_file"])
