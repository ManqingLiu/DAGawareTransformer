import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from src.models.DAG_aware_transformer import TabularBERT
from src.models.utils import generate_dag_edges, ModelTrainer, rmse, IPTW_stabilized, AIPW, set_seed, elastic_net_penalty
from src.data.data_preprocess import DataProcessor
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import expit as sigmoid  # Import sigmoid function
from config import *
import wandb


set_seed()

##### Part I: data pre-processing #####
n_sample = N_SAMPLE

dataset_type = 'twins'  # Can be 'cps' or 'psid', depending on what you want to use

# Use the variable in the file path
#dataframe = pd.read_csv(f'data/realcause_datasets/lalonde_{dataset_type}_sample{n_sample}.csv')
dataframe = pd.read_csv(f'data/realcause_datasets/{dataset_type}_sample{n_sample}.csv')

# get true ATE: mean of y1 - mean of y0
ATE_true = dataframe['y1'].mean() - dataframe['y0'].mean()
print("true ATE:", ATE_true)
# remove last 3 columns of the dataframe
df = dataframe.iloc[:, :-3].copy()
# dataframe = dataframe.iloc[:, :-3]
num_bins = NUM_BINS
processor = DataProcessor(df)
processor.sample_variables()
processor.bin_continuous_variables(num_bins)
tensor, feature_names = processor.create_tensor()
binary_dims, continuous_dims = processor.generate_dimensions()
binary_features, _ = processor.get_feature_names()  # Get binary and continuous feature names
dag = generate_dag_edges(feature_names)
batch_size = 32
test_size = TEST_SIZE
# Split data and create DataLoaders

train_loader, train_data, val_loader, val_data, \
val_loader_A1, val_data_A1, val_loader_A0, val_data_A0, \
train_loader_A1, train_data_A1, train_loader_A0, train_data_A0 = (
    processor.split_data_loaders(tensor, batch_size=batch_size, test_size=test_size, random_state=SEED_VALUE,
                                 feature_names=feature_names))



###### Part II: model training and validation ######
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))


# Create the model
num_nodes = len(feature_names)  # Total number of nodes in the graph
embedding_dim = 128  # Embedding dimension
nhead = 4  # Number of heads in multihead attention
learning_rate = LEARNING_RATE # Learning rate
n_epochs = N_EPOCHS  # Number of epochs



# Instantiate the model, optimizer, and loss function
model = TabularBERT(num_nodes=num_nodes, embedding_dim=embedding_dim, nhead=nhead,
                    categorical_dims=binary_dims, continuous_dims=continuous_dims,
                    dag=dag, batch_size=batch_size, device=device, dropout_rate=DROPOUT_RATE)

# model.load_state_dict(torch.load(model_path))
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
criterion_binary = nn.BCEWithLogitsLoss()
criterion_continuous = nn.MSELoss()


trainer = ModelTrainer(model, train_loader, val_loader, binary_features,
                       feature_names, criterion_binary, criterion_continuous, device)

# Train the model
# Assume the rest of your setup (model, optimizer, etc.) is already defined above

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
        project="dag-aware-transformer-gformula-IPTW-AIPW",
        #name=f"lalonde_{dataset_type} data (sample {n_sample})",
        name=f"{dataset_type} data (sample {n_sample})",

        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": "Transformer",
            #"dataset": f"lalonde_{dataset_type} data (sample {n_sample})",
            "dataset": f"{dataset_type} data (sample {n_sample})",
            "weight decay": f"{WEIGHT_DECAY}",
            "dropout rate": f"{DROPOUT_RATE}",
            "loader": f"{loader}",
            "epochs": n_epochs,
        }
    )

    for epoch in range(n_epochs):
        # Train on the current loader
        train_losses = trainer.train(optimizer, loader)

        # Log the losses
        if epoch % 1 == 0:  # Logging frequency
            train_loss_strings = [f'{feature}: {loss:.5f}' for feature, loss in train_losses.items()]
            print(f'{loader_name.capitalize()} Loader, Epoch {epoch}, Training Losses: {", ".join(train_loss_strings)}')
            # Replace wandb.log with your preferred logging method if necessary
            wandb.log({f'{loader_name.capitalize()} Training Loss - {feature}': loss for feature, loss in
                       train_losses.items()}, step=epoch)
    wandb.finish()
    # Save the model after training on each loader
    model_path = f"experiments/model/model_{dataset_type}_{loader_name}_sample{n_sample}_epoch{n_epochs}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model trained on {loader_name} data saved to {model_path}")
