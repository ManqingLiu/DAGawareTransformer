import ray
import os
import tempfile
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
import torch
import torch.nn as nn
from torch.optim import AdamW
from src.models.DAG_aware_transformer import TabularBERT
from src.models.utils import *
from src.data.data_preprocess import DataProcessor
from config import *
import wandb

dataset_type = 'twins'

# Use the variable in the file path
dataframe = pd.read_csv(f'data/realcause_datasets/{dataset_type}_sample{N_SAMPLE}.csv')

df = dataframe.iloc[:, :-3].copy()
num_bins = NUM_BINS
processor = DataProcessor(df)
processor.sample_variables()
processor.bin_continuous_variables(num_bins)
tensor, feature_names = processor.create_tensor()
binary_dims, continuous_dims = processor.generate_dimensions()
binary_features, _ = processor.get_feature_names()  # Get binary and continuous feature names
dag = generate_dag_edges(feature_names)


def train_tabular_bert(config, checkpoint_dir=None, data_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, train_data, val_loader, val_data, \
        val_loader_A1, val_data_A1, val_loader_A0, val_data_A0, \
        train_loader_A1, train_data_A1, train_loader_A0, train_data_A0 = (
        processor.split_data_loaders(tensor, batch_size=config["batch_size"], test_size=TEST_SIZE, random_state=SEED_VALUE,
                                     feature_names=feature_names))

    model = TabularBERT(num_nodes=len(feature_names),
                        embedding_dim=config["embedding_dim"],
                        nhead=config["n_head"],
                        categorical_dims=binary_dims,
                        continuous_dims=continuous_dims,
                        dag=dag,
                        batch_size=config["batch_size"],
                        device=device,
                        dropout_rate=config["dropout_rate"]).to(device)

    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion_binary = nn.BCEWithLogitsLoss()
    criterion_continuous = nn.MSELoss()

    trainer = ModelTrainer(model, train_loader, val_loader, binary_features,
                           feature_names, criterion_binary, criterion_continuous, device)

    # Train the model
    loaders = {
        "train": train_loader,
        "val": val_loader
    }

    cumulative_t_hat_loss_train = 0
    cumulative_t_hat_loss_val = 0
    num_loaders = len(loaders)

    n_epochs = config["n_epochs"]
    LOG_FEQ = n_epochs // 5

    for epoch in range(n_epochs):
        for loader_name, loader in loaders.items():
            # Train on the current loader
            train_losses = trainer.train(optimizer, loader)

            # Log the losses
            if epoch % LOG_FEQ == 0:
                train_loss_strings = [f'{feature}: {loss:.5f}' for feature, loss in train_losses.items()]
                print(f'{loader_name.capitalize()} Loader, Epoch {epoch}, Training Losses: {", ".join(train_loss_strings)}')

            # Switch the loader for validation based on current loader_name
            validation_loader = val_loader if loader_name == "train" else train_loader

            val_losses = trainer.validate(validation_loader)
            if 't_hat' in val_losses:
                t_hat_loss = val_losses['y_hat']
                if loader_name == "train":
                    cumulative_t_hat_loss_train += t_hat_loss
                else:
                    cumulative_t_hat_loss_val += t_hat_loss

                if epoch % LOG_FEQ == 0:
                    print(f'{loader_name.capitalize()} Loader, Epoch {epoch}, t_hat Validation Loss: {t_hat_loss:.5f}')

        # After going through both loaders, calculate and report the average validation loss
        if epoch % LOG_FEQ == 0 or epoch == n_epochs - 1:
            avg_t_hat_loss = (cumulative_t_hat_loss_train + cumulative_t_hat_loss_val) / num_loaders
            # Report the average validation loss to Ray Tune
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                torch.save(
                    (model.state_dict(), optimizer.state_dict()), path
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                train.report({"loss": avg_t_hat_loss}, checkpoint = checkpoint)

        # Reset cumulative validation losses after each epoch to prepare for the next epoch's calculations
        cumulative_t_hat_loss_train = 0
        cumulative_t_hat_loss_val = 0

        # Define the hyperparameter search space

config = {
    "n_epochs": tune.choice([10, 20]),
    #"lr": tune.loguniform(1e-3, 1e-4),
    "lr": tune.choice([1e-3, 1e-4]),
    "batch_size": tune.choice([32, 64, 128]),
    #"weight_decay": tune.uniform(0, 0.03),
    "weight_decay": tune.choice([0.05, 0.1]),
    "embedding_dim": tune.choice([256, 512]),
    "n_head": tune.choice([16, 32]),
    #"dropout_rate": tune.uniform(0, 0.4)
    "dropout_rate": tune.choice([0.1, 0.2])
}


# Optionally, define a scheduler and a search algorithm
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=20,
    grace_period=1,
    reduction_factor=2)

# Start the Ray Tune run
analysis = tune.run(
    train_tabular_bert,
    resources_per_trial={"cpu": 1, "gpu": 1 if torch.cuda.is_available() else 0},
    config=config,
    num_samples=1,  # Number of times to sample from the hyperparameter space
    scheduler=scheduler
)

best_trial = analysis.get_best_trial("loss", "min", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final t_hat validation loss: {}".format(best_trial.last_result["loss"]))
