from argparse import ArgumentParser
import json
import os
import time

import pandas as pd
import torch
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader
import wandb

from src.dataset import CausalDataset
from src.models.transformer_model import DAGTransformer
from src.predict import predict
from src.train.lalonde_psid.train import train
from src.train.lalonde_psid.train_metrics import make_gifs
from src.utils import replace_column_values


parser = ArgumentParser()
parser.add_argument(
    "--config", type=str, required=True, default="config/train/lalonde_psid.json"
)
args = parser.parse_args()

with open(args.config) as f:
    config = json.load(f)

filepaths = config["filepaths"]
train_config = config["training"]
model_config = config["model"]
random_seed = config["random_seed"]
torch.manual_seed(random_seed)

with open(filepaths["dag"]) as f:
    dag = json.load(f)

# Move all of this to a utils function for load_dag and load_data
num_nodes = len(dag["nodes"])
dag["node_ids"] = dict(zip(dag["nodes"], range(num_nodes)))
print(dag)

model = DAGTransformer(dag=dag, **model_config)

train_data = pd.read_csv(filepaths["data_train_file"])
train_data_model = train_data[dag["nodes"]]
train_dataset = CausalDataset(train_data_model, dag, random_seed)

batch_size = train_config["batch_size"]
train_dataloader = DataLoader(
    train_dataset,
    sampler=ImbalancedDatasetSampler(train_dataset),
    batch_size=batch_size,
    collate_fn=train_dataset.collate_fn,
)

val_data = pd.read_csv(filepaths["data_val_file"])
val_data_model = val_data[dag["nodes"]]
val_dataset = CausalDataset(val_data_model, dag, random_seed)
val_dataloader = DataLoader(
    val_dataset,
    sampler=ImbalancedDatasetSampler(val_dataset),
    batch_size=batch_size,
    collate_fn=val_dataset.collate_fn,
)

test_data = pd.read_csv(filepaths["data_test_file"])
test_data_model = test_data[dag["nodes"]]
test_dataset = CausalDataset(test_data_model, dag, random_seed)
test_dataloader = DataLoader(
    test_dataset,
    sampler=ImbalancedDatasetSampler(test_dataset),
    batch_size=batch_size,
    collate_fn=test_dataset.collate_fn,
)

run = wandb.init(project="DAG transformer", entity="mliu7", config=config)
start_time = time.time()
model = train(
    model,
    val_data,
    train_data,
    dag,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    train_config,
    random_seed=random_seed,
)
print("Done training.")
wandb.finish()

make_gifs(run)

predictions = predict(
    model,
    val_data,
    dag,
    val_dataloader,
    mask=train_config["dag_attention_mask"],
    random_seed=random_seed,
)

data_A1 = replace_column_values(val_data, "t", 1)
dataset_A1 = CausalDataset(data_A1, dag, random_seed)
dataloader_A1 = DataLoader(
    dataset_A1,
    sampler=ImbalancedDatasetSampler(dataset_A1),
    batch_size=batch_size,
    collate_fn=val_dataset.collate_fn,
)

predictions_A1 = predict(
    model,
    data_A1,
    dag,
    dataloader_A1,
    mask=train_config["dag_attention_mask"],
    random_seed=random_seed,
)

# rename pred_y to pred_y_A1
predictions_A1 = predictions_A1.rename(columns={"pred_y": "pred_y_A1"})

data_A0 = replace_column_values(val_data, "t", 0)
dataset_A0 = CausalDataset(data_A0, dag, random_seed)
dataloader_A0 = DataLoader(
    dataset_A0,
    sampler=ImbalancedDatasetSampler(dataset_A0),
    batch_size=batch_size,
    collate_fn=val_dataset.collate_fn,
)
predictions_A0 = predict(
    model,
    data_A0,
    dag,
    dataloader_A0,
    mask=train_config["dag_attention_mask"],
    random_seed=random_seed,
)

# rename pred_y to pred_y_A0
predictions_A0 = predictions_A0.rename(columns={"pred_y": "pred_y_A0"})

# create final predictions train data including prob_t from predictions_train, pred_y_A1 from predictions_train_A1,
# and pred_y_A0 from predictions_train_A0
final_predictions = pd.concat(
    [predictions["t_prob"], predictions_A1["pred_y_A1"], predictions_A0["pred_y_A0"]],
    axis=1,
)

output_dir = os.path.dirname(filepaths["output_file"])
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the predictions to a CSV file
final_predictions.to_csv(filepaths["output_file"], index=False)

# After training for holdout file
end_time = time.time()

# Calculate and print the total wall time
total_wall_time = end_time - start_time
# Convert the total wall time to minutes and seconds
minutes, seconds = divmod(total_wall_time, 60)
print(f"Total wall time used: {minutes} minutes and {seconds} seconds")
