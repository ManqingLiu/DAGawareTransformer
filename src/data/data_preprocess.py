import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.dataset import CausalDataset
def data_preprocess(config, filepaths, dag):
    train_config = config["training"]
    random_seed = config["random_seed"]
    torch.manual_seed(random_seed)

    num_nodes = len(dag["nodes"])
    dag["node_ids"] = dict(zip(dag["nodes"], range(num_nodes)))

    train_data = pd.read_csv(filepaths["data_train_file"])
    train_data_model = train_data[dag["nodes"]]
    train_dataset = CausalDataset(train_data_model, dag, random_seed)

    batch_size = train_config["batch_size"]
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    val_data = pd.read_csv(filepaths["data_val_file"])
    val_data_model = val_data[dag["nodes"]]
    val_dataset = CausalDataset(val_data_model, dag, random_seed)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=val_dataset.collate_fn,
    )

    test_data = pd.read_csv(filepaths["data_test_file"])
    test_data_model = test_data[dag["nodes"]]
    test_dataset = CausalDataset(test_data_model, dag, random_seed)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=test_dataset.collate_fn,
    )

    return train_data, train_dataloader, val_data, val_dataloader, test_data, test_dataloader

