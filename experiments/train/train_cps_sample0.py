import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from src.models.DAG_aware_transformer import *
from src.models.utils import *
from src.dataset.data_preprocess import DataProcessor
# TODO: Convert config to a json file
from config import *
import wandb

set_seed()
##### Part I: data pre-processing #####
# TODO: Convert to config files and command line arguments
dataset_type = 'lalonde_psid'  # Can be 'cps' or 'psid', depending on what you want to use

# Use the variable in the file path
dataframe = pd.read_csv(f'data/realcause_datasets/{dataset_type}_sample{N_SAMPLE}.csv')

df = dataframe.iloc[:, :-3].copy()
num_bins = NUM_BINS
processor = DataProcessor(df)
#processor.sample_variables()
processor.bin_continuous_variables(num_bins)
tensor, feature_names = processor.create_tensor()
# TODO: This is due to the list of embeddings
binary_dims, continuous_dims = processor.generate_dimensions()
binary_features, _ = processor.get_feature_names()  # Get binary and continuous feature names
#dag = generate_dag_edges(feature_names)
print(feature_names)
print(dag)

# TODO Refactor this to several explicit steps
train_loader, train_data, val_loader, val_data, \
val_loader_A1, val_data_A1, val_loader_A0, val_data_A0, \
train_loader_A1, train_data_A1, train_loader_A0, train_data_A0 = (
    processor.split_data_loaders(tensor, batch_size=BATCH_SIZE, test_size=TEST_SIZE, random_state=SEED_VALUE,
                                 feature_names=feature_names))


'''
# print first batch in val_loader
for i, batch in enumerate(val_loader):
    print(i, batch)
    break
'''

###### Part II: model training and validation ######
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the model
num_nodes = len(feature_names)  # Total number of nodes in the graph


# Instantiate the model, optimizer, and loss function
model = TabularBERT(num_nodes=num_nodes,
                    embedding_dim=EMBEDDING_DIM,
                    nhead=N_HEAD,
                    dag=dag,
                    batch_size=BATCH_SIZE,
                    categorical_dims=binary_dims,
                    continuous_dims=continuous_dims,
                    device=device,
                    dropout_rate=DROPOUT_RATE)

# num_nodes, embedding_dim, nhead, dag, batch_size, categorical_dims, continuous_dims, device, dropout_rate

# model.load_state_dict(torch.load(model_path))
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion_binary = nn.BCEWithLogitsLoss()

# TODO: Investigate the loss function for continuous variables
criterion_continuous = nn.MSELoss()

trainer = ModelTrainer(model,
                       train_loader, val_loader, binary_features,
                           feature_names, criterion_binary, criterion_continuous, device)

# Train the model
loaders = {
    "train": train_loader,
    "val": val_loader
}


for loader_name, loader in loaders.items():

    if wandb.run is not None:
        wandb.finish()
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"dag-aware-transformer-{dataset_type}",
        #name=f"lalonde_{dataset_type} data (sample {n_sample})",
        name=f"{dataset_type} data (sample {N_SAMPLE})",

        # track hyperparameters and run metadata
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "Transformer",
            #"dataset": f"lalonde_{dataset_type} data (sample {n_sample})",
            "dataset": f"{dataset_type} data (sample {N_SAMPLE})",
            "masking": "without masking",
            "weight decay": f"{WEIGHT_DECAY}",
            "dropout rate": f"{DROPOUT_RATE}",
            "batch size": f"{BATCH_SIZE}",
            "embedding dimension": f"{EMBEDDING_DIM}",
            "nhead": f"{N_HEAD}",
            "loader": f"{loader}",
            "epochs": N_EPOCHS,
        }
    )

    # Switch the loader for validation based on current loader_name
    validation_loader = val_loader if loader_name == "train" else train_loader

    for epoch in range(N_EPOCHS):
        # Train on the current loader
        train_losses = trainer.train(optimizer, loader)

        # Log the losses
        if epoch % LOG_FEQ == 0:
            train_loss_strings = [f'{feature}: {loss:.5f}' for feature, loss in train_losses.items()]
            print(f'{loader_name.capitalize()} Loader, Epoch {epoch}, Training Losses: {", ".join(train_loss_strings)}')
            # Replace wandb.log with your preferred logging method if necessary
            wandb.log({f'{loader_name.capitalize()} Training Loss - {feature}': loss for feature, loss in
                       train_losses.items()}, step=epoch)

        # Calculate the validation losses
        val_losses = trainer.validate(validation_loader)
        #print(val_losses)
        if epoch % LOG_FEQ == 0:
            # Log the validation losses
            val_loss_strings = [f'{feature}: {loss:.5f}' for feature, loss in val_losses.items()]
            print(f'{loader_name.capitalize()} Loader, Epoch {epoch}, Validation Losses: {", ".join(val_loss_strings)}')
            wandb.log({f'{loader_name.capitalize()} Validation Loss - {feature}': loss for feature, loss in
                           val_losses.items()}, step=epoch)

    wandb.finish()
    # Save the model after training on each loader

    #model_path = f"experiments/model/{dataset_type}/model_{dataset_type}_{loader_name}_sample{N_SAMPLE}_epoch{N_EPOCHS}.pth"
    #torch.save(model.state_dict(), model_path)
    save_model(model, dataset_type, loader_name, N_SAMPLE, N_EPOCHS,
               LEARNING_RATE, WEIGHT_DECAY, DROPOUT_RATE, BATCH_SIZE, EMBEDDING_DIM, N_HEAD)
    #print(f"Model trained on {loader_name} data saved to {model_path}")
