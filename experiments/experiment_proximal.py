from typing import Dict, Any, Optional
import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
from typing import Dict, Any
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from src.utils_proximal import grid_search_dict
from src.models.NMMR.NMMR_experiments import NMMR_experiment
logger = logging.getLogger()


def get_run_func(mdl_name: str):
    if mdl_name == "nmmr":
        return NMMR_experiment
    else:
        raise ValueError(f"name {mdl_name} is not known")


def experiments(configs: Dict[str, Any],
                dag: Dict[str, Any],
                dump_dir: Path,
                num_cpus: int):
    data_config = configs["data"]
    model_config = configs["model"]
    train_config = configs['training']
    #dag = configs_dag
    num_nodes = len(dag['nodes'])
    dag['node_ids'] = dict(zip(dag['nodes'], range(num_nodes)))
    mask = True
    n_repeat: int = configs["n_repeat"]

    if num_cpus <= 1 and n_repeat <= 1:
        verbose: int = 2
    else:
        verbose: int = 0

    run_func = get_run_func(model_config["name"])
    for dump_name, env_param in grid_search_dict(data_config):
        one_dump_dir = dump_dir.joinpath(dump_name)
        os.mkdir(one_dump_dir)
        for mdl_dump_name, mdl_param in grid_search_dict(model_config):
            if mdl_dump_name != "one":
                one_mdl_dump_dir = one_dump_dir.joinpath(mdl_dump_name)
                os.mkdir(one_mdl_dump_dir)
            else:
                one_mdl_dump_dir = one_dump_dir

            if model_config.get("log_metrics", False) == "True":
                test_losses = []
                train_metrics_ls = []
                for idx in range(n_repeat):
                    test_loss, train_metrics = run_func(
                        env_param, mdl_param, train_config, dag, mask, one_mdl_dump_dir, idx, verbose)
                    train_metrics['rep_ID'] = idx
                    train_metrics_ls.append(train_metrics)
                    if test_loss is not None:
                        test_losses.append(test_loss)

                if test_losses:
                    np.savetxt(one_mdl_dump_dir.joinpath("result.csv"), np.array(test_losses))
                metrics_df = pd.concat(train_metrics_ls).reset_index()
                metrics_df.rename(columns={'index': 'epoch_num'}, inplace=True)
                metrics_df.to_csv(one_mdl_dump_dir.joinpath("train_metrics.csv"), index=False)
            else:
                test_losses = []
                for idx in range(n_repeat):
                    test_loss = run_func(
                        env_param, mdl_param, train_config, dag, mask, one_mdl_dump_dir, idx, verbose)
                    if test_loss is not None:
                        test_losses.append(test_loss)
                if test_losses:
                    np.savetxt(one_mdl_dump_dir.joinpath("result.csv"), np.array(test_losses))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dag', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--results_dir', type=str, required=True)
    # Load the configurations from the JSON file
    args = parser.parse_args()

    with open(args.dag) as f:
        print(f'Loading dag file from {args.dag}')
        dag = json.load(f)

    with open(args.config) as f:
        print(f'Loading config file from {args.config}')
        config = json.load(f)

    # Define the directory where the model will be saved
    dump_dir = Path(args.results_dir)

    # Define the number of CPUs to use
    num_cpus = 20  # replace with the number of CPUs you want to use

    # Run the experiments
    experiments(config, dag, dump_dir, num_cpus)