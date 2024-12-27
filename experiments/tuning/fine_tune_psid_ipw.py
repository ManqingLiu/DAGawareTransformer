import os
import json
from argparse import ArgumentParser

import torch
from ray import train, tune

from experiments.tuning.config import config_psid_ipw
from src.utils import log_results
from src.data.lalonde import data_preprocess
from src.experiment import experiment


def fine_tune(config, dag, train_dataloader, val_dataloader, val_data):
    best_ipw_rmse = float("inf")
    best_ipw_epoch = 0

    for epoch in range(config['training']['num_epochs']):
        model_trained, train_loss, val_loss, predictions = experiment(config, dag,
                                                                      train_dataloader,
                                                                      val_dataloader, val_data, random_seed=False)
        std_rmse_ = std_rmse(predictions['pred_y_A0'], predictions['pred_y_A1'])
        ipw_rmse_ = ipw_rmse(predictions['y'], predictions['t'], predictions['t_prob'])
        cfcv_rmse_ = cfcv_rmse(predictions['y'], predictions['t'], predictions['pred_y_A0'],
                            predictions['pred_y_A1'], predictions['t_prob'])

        if ipw_rmse_ < best_ipw_rmse:
            best_ipw_rmse = ipw_rmse_
            best_ipw_epoch = epoch

        train.report({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "std_rmse":  std_rmse_,
                      "ipw_rmse": ipw_rmse_, "cfcv_rmse": cfcv_rmse_, "best_ipw_rmse": best_ipw_rmse,
                      "best_ipw_epoch": best_ipw_epoch})



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
        config=config_psid_ipw,
        num_samples=10,  # Number of times to sample from the hyperparameter space
        resources_per_trial={"cpu": 20, "gpu": 4 if torch.cuda.is_available() else 0},
        metric="best_ipw_rmse",
        mode="min"
    )

    best_trial_ipw = analysis.get_best_trial("best_ipw_rmse", "min", "all")
    print("Best trial config: {}".format(best_trial_ipw.config))
    print("Best trial final validation IPW RMSE: {}".format(best_trial_ipw.last_result["best_ipw_rmse"]))
    print("Best trial epoch: {}".format(best_trial_ipw.last_result["best_ipw_epoch"]))

    # Log the results
    log_results(best_trial_ipw.config, filepaths["result_file_ipw"], best_trial_ipw.last_result["best_ipw_epoch"])