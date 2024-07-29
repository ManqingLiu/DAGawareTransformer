import os
import json
from argparse import ArgumentParser

import torch
from ray import train, tune

from experiments.tuning.config import config_cps_dr
from src.utils import log_results
from src.data.lalonde import data_preprocess
from src.experiment import experiment
from src.train.lalonde_cps.train_metrics import std_rmse, ipw_rmse, cfcv_rmse


def fine_tune(config, dag, train_dataloader, val_dataloader, val_data):
    best_cfcv_rmse = float("inf")
    best_cfcv_epoch = 0

    for epoch in range(config['training']['num_epochs']):
        model_trained, train_loss, val_loss, predictions = experiment(config, dag,
                                                                      train_dataloader,
                                                                      val_dataloader, val_data, random_seed=False)
        std_rmse_ = std_rmse(predictions['pred_y_A0'], predictions['pred_y_A1'])
        ipw_rmse_ = ipw_rmse(predictions['y'], predictions['t'], predictions['t_prob'])
        cfcv_rmse_ = cfcv_rmse(predictions['y'], predictions['t'], predictions['pred_y_A0'],
                            predictions['pred_y_A1'], predictions['t_prob'])

        if cfcv_rmse_ < best_cfcv_rmse:
            best_cfcv_rmse = cfcv_rmse_
            best_cfcv_epoch = epoch

        train.report({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "std_rmse":  std_rmse_,
                      "ipw_rmse": ipw_rmse_, "cfcv_rmse": cfcv_rmse_, "best_cfcv_rmse": best_cfcv_rmse,
                      "best_cfcv_epoch": best_cfcv_epoch})


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True
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
        config=config_cps_dr,
        num_samples=10,  # Number of times to sample from the hyperparameter space
        resources_per_trial={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0},
        metric="best_cfcv_rmse",
        mode="min"
    )

    best_trial_cfcv = analysis.get_best_trial("best_cfcv_rmse", "min", "all")
    print("Best trial config: {}".format(best_trial_cfcv.config))
    print("Best trial final validation CFCV RMSE: {}".format(best_trial_cfcv.last_result["best_cfcv_rmse"]))
    print("Best trial epoch: {}".format(best_trial_cfcv.last_result["best_cfcv_epoch"]))

    # Log the results
    log_results(best_trial_cfcv.config, filepaths["result_file_cfcv"], best_trial_cfcv.last_result["best_cfcv_epoch"])