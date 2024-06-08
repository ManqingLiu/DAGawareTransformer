import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
import json
from src.train.lalonde_psid.train import train as train_function
from src.model import *
import pandas as pd
from src.dataset import CausalDataset
from torch.utils.data import DataLoader
from config import *
import torch
from argparse import ArgumentParser
from src.utils import log_results
import os


def fine_tune(config, dag, train_data, holdout_data):
    train_data = train_data[dag['nodes']]
    train_dataset = CausalDataset(train_data, dag, random_seed=config['random_seed'])
    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=train_dataset.collate_fn)

    # Create the model
    model = DAGTransformer(dag=dag, **config['model'])

    # Train the model using the train function from train.py
    train_function(model, train_dataloader, config['training'], mask=True, save_model=False, model_file=None)

    # Evaluate the model on the holdout set and report the loss
    holdout_data = holdout_data[dag['nodes']]
    holdout_dataset = CausalDataset(holdout_data, dag, random_seed=config['random_seed'])
    holdout_dataloader = DataLoader(holdout_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=holdout_dataset.collate_fn)

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    with torch.no_grad():
        holdout_loss = 0
        for batch in holdout_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch, mask=True)
            batch_loss, _ = causal_loss_fun(outputs, batch)
            holdout_loss += batch_loss.item()
        avg_holdout_loss = holdout_loss / len(holdout_dataloader)


        train.report({"loss":  avg_holdout_loss})



if __name__ == '__main__':
    current_directory = os.getcwd()
    print(f"Current directory: {current_directory}")

    parser = ArgumentParser()
    parser.add_argument('--dag', type=str, required=True)
    parser.add_argument('--data_train_file', type=str, required=True)
    parser.add_argument('--data_holdout_file', type=str, required=True)
    parser.add_argument('--results', type=str, required=True)
    args = parser.parse_args()

    with open(args.dag) as f:
        print(f'Loading dag file from {args.dag}')
        dag = json.load(f)

    train_data = pd.read_csv(args.data_train_file)
    holdout_data = pd.read_csv(args.data_holdout_file)

    # Wrap the train_tabular_bert function with the dag argument
    fine_tune_new = tune.with_parameters(fine_tune, dag=dag, train_data=train_data, holdout_data=holdout_data)


    analysis = tune.run(
        fine_tune_new,
        config=config_twins,
        num_samples=10,  # Number of times to sample from the hyperparameter space
        resources_per_trial={"cpu": 1, "gpu": 1 if torch.cuda.is_available() else 0},
    )

    best_trial = analysis.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    # Log the results
    log_results(best_trial.config, args.results)
