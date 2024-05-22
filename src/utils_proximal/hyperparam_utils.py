import argparse
import json
import os
import os.path as op

import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator

from src.utils import grid_search_dict


def get_hyperparameter_results_dataframe(dump_dir):
    config_path = os.path.join(dump_dir, 'configs.json')
    with open(config_path) as config_file:
        config = json.load(config_file)
    data_config = config["data"]
    model_config = config["model"]

    results_df = pd.DataFrame()

    for dump_name, env_param in grid_search_dict(data_config):
        one_dump_dir = os.path.join(dump_dir, dump_name)
        for mdl_dump_name, mdl_param in grid_search_dict(model_config):
            if mdl_dump_name != "one":
                one_mdl_dump_dir = os.path.join(one_dump_dir,mdl_dump_name)
            else:
                one_mdl_dump_dir = one_dump_dir
            combined_param_dict = env_param | mdl_param
            if os.path.exists(one_mdl_dump_dir):
                # tensorboard_dir = os.path.join(one_mdl_dump_dir, 'tensorboard_log')
                df = pd.read_csv(os.path.join(one_mdl_dump_dir, 'train_metrics.csv'))
                max_avg_causal_val_loss = df.groupby('rep_ID').mean().obs_MSE_val.max()

                # n_epochs = combined_param_dict['n_epochs']
                # for tensorboard_logfile in os.listdir(tensorboard_dir):
                #     tensorboard_filepath = os.path.join(tensorboard_dir, tensorboard_logfile)
                #     temp_df = pd.DataFrame([combined_param_dict])
                #     for event in summary_iterator(tensorboard_filepath):
                #         if event.step == (n_epochs-1):
                #             temp_df['causal_loss_val'] = event.summary.value[0].simple_value

                combined_param_dict['max_avg_val_loss'] = max_avg_causal_val_loss
                temp_df = pd.DataFrame([combined_param_dict])
                results_df = pd.concat([results_df, temp_df])

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_dir', default='/Users/kompa/Downloads/naive_neural_net_AWY_08-01-17-51-42')
    parser.add_argument('--out_dir', default='/Users/kompa/Downloads/')
    parser.add_argument('--experiment')
    args = parser.parse_args()

    results_df = get_hyperparameter_results_dataframe(args.dump_dir)
    results_df.to_csv(op.join(args.out_dir, f"hp_results_{os.path.basename(os.path.normpath(args.dump_dir))}.csv"), index=False)
    # results_df.to_pickle(op.join(args.out_dir, "hp_results.pkl"))
