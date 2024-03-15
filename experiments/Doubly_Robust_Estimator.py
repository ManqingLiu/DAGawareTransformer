import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.DAG_aware_transformer import TabularBERT
from src.models.utils import get_predicted_probabilities, validate, AIPW

##### Part I: data pre-processing #####
n_samples = 5000
file_path_nointeraction = f'data/raw/simulation{n_samples}_nointeraction.csv'
file_path_interaction = f'data/raw/simulation{n_samples}_interaction.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path_nointeraction)
new_order = ['sex', 'sex_hat', 'asthma', 'asthma_hat', 'treatment', 'treatment_hat', 'outcome', 'outcome_hat',
             'age_discretized', 'age_hat_discretized', 'smoke_intensity_discretized', 'smoke_intensity_hat_discretized']

df = df[new_order]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train and est split
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

train_data, estimation_data = train_test_split(df, test_size=0.5, random_state=42)
train_tensor = torch.tensor(train_data.values, dtype=torch.long)
estimation_tensor = torch.tensor(estimation_data.values, dtype=torch.long)

# Create the model
num_nodes = 12  # Total number of nodes in the graph
embedding_dim = 128  # Embedding dimension
nhead = 4  # Number of heads in multihead attention
dag = [(0, 5), (0, 7), (2, 5), (2, 7), (4, 7), (8, 5), (8, 7), (10, 5), (10, 7)]  # Example DAG
categorical_dims = [2, 2, 2, 2, 2, 2, 2, 2]  # Binary features
continuous_dims = [20, 20, 20, 20]  # Continuous features, discretized into 10 bins

# Create DataLoader objects for the training and estimation sets
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=32, shuffle=True)
estimation_loader = DataLoader(TensorDataset(estimation_tensor), batch_size=32)

binary_features = ['sex', 'sex_hat', 'asthma', 'asthma_hat', 'treatment', 'treatment_hat', 'outcome', 'outcome_hat']
continuous_features = ['age_discretized', 'age_hat_discretized', 'smoke_intensity_discretized', 'smoke_intensity_hat_discretized']
features = ['sex', 'asthma','treatment','outcome','age_discretized', 'smoke_intensity_discretized']


###### Part II: Cross-fitting  ######
#### Round I: use train data to get estimators
# Instantiate the model, optimizer, and loss function
model_train = TabularBERT(num_nodes=num_nodes, embedding_dim=128, nhead=4, dag=dag,
                    categorical_dims=categorical_dims, continuous_dims=continuous_dims)
mode_train = model_train.to(device)
learning_rate = 1e-4
optimizer_train = optim.Adam(model_train.parameters(), lr=learning_rate)
criterion_binary = nn.BCEWithLogitsLoss()
criterion_continuous = nn.MSELoss()


import wandb

n_epochs = 100

if wandb.run is not None:
    wandb.finish()
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="dag-aware-transformer-crossfit-AIPW",
    name = "use train data to train",

    # track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "architecture": "Transformer",
        "dataset": f"simulation with {n_samples} samples without interaction terms",
        "epochs": n_epochs,
    }
)

# Training phase
model_train.train()
torch.manual_seed(287)
for epoch in range(n_epochs):
    for batch in train_loader:  # Process each batch
        batch_gpu = batch[0].to(device)  # Assuming each batch has only data, no labels

        optimizer_train.zero_grad()
        losses = {}
        num_feature_pairs = len(binary_features + continuous_features)  # Assuming binary_features includes both original and _hat features
        feature_names = binary_features + continuous_features

        for i in range(0, num_feature_pairs - 1, 2):
            feature_name = feature_names[i]
            feature_name_hat = feature_name + '_hat'

            output = model_train(batch_gpu)[:, i + 1].unsqueeze(1)
            true_values = batch_gpu[:, i].unsqueeze(1)

            if feature_name in binary_features:
                criterion = criterion_binary
            else:
                criterion = criterion_continuous

            loss = criterion(output.float(), true_values.float())
            loss.backward(retain_graph=True)
            losses[feature_name_hat] = loss.item()

        optimizer_train.step()
        optimizer_train.zero_grad()


    if epoch % 10 == 0:
        train_loss_strings = [f'{feature}: {loss:.5f}' for feature, loss in losses.items()]
        print(f'Epoch {epoch}, Training Losses: {", ".join(train_loss_strings)}')
        wandb.log({f'Training Loss - {feature}': loss for feature, loss in losses.items()}, step=epoch)

wandb.finish()

feature_to_index = {'sex': 0, 'sex_hat': 1, 'asthma': 2, 'asthma_hat': 3, 'treatment': 4, 'treatment_hat': 5,
                    'outcome': 6, 'outcome_hat': 7,
                    'age_discretized': 8, 'age_hat_discretized': 9, 'smoke_intensity_discretized': 10,
                    'smoke_intensity_hat_discretized': 11}

# Now you can use the dictionary to reference the 'treatment' column
treatment_column_index = feature_to_index['treatment']

#### Y_a1
# Set all values in the 'treatment' column to 1
train_data_A1 = train_tensor.clone()
train_data_A1[:, treatment_column_index] = 1

#### A. get probabilities using training data
probabilities_A1 = get_predicted_probabilities(model_train, train_data_A1)

#### B. get b_hat (outcome_hat) and pi_hat (treatment_hat)
outcome_hat_train = probabilities_A1['outcome_hat']
treatment_hat_train = probabilities_A1['treatment_hat']

A1 = np.ones(len(estimation_data))
Y_a1_est = AIPW(A1, treatment_hat_train, estimation_data['outcome'], outcome_hat_train)
print(f"Y_a1_est:{Y_a1_est}")

#### Y_a0
# Set all values in the 'treatment' column to 0
train_data_A0 = train_tensor.clone()
train_data_A0[:, treatment_column_index] = 0


#### A. get probabilities using training data
probabilities_A0 = get_predicted_probabilities(model_train, train_data_A0)

#### B. get b_hat (outcome_hat) and pi_hat (treatment_hat)
outcome_hat_train = probabilities_A0['outcome_hat']
treatment_hat_train = probabilities_A0['treatment_hat']

A0 = np.zeros(len(estimation_data))
Y_a0_est = AIPW(A0, treatment_hat_train, estimation_data['outcome'], outcome_hat_train)
print(f"Y_a0_est:{Y_a0_est}")

Phi_1 = Y_a1_est-Y_a0_est
print(f"Phi_1: {Phi_1}")


#### Round II: use estimation data to get estimators
model_est = TabularBERT(num_nodes=num_nodes, embedding_dim=128, nhead=4, dag=dag,
                    categorical_dims=categorical_dims, continuous_dims=continuous_dims)
model_est = model_est.to(device)

if wandb.run is not None:
    wandb.finish()
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="dag-aware-transformer-crossfit-AIPW",
    name = "use estimation data to train",

    # track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "architecture": "Transformer",
        "dataset": f"simulation with {n_samples} samples without interaction terms",
        "epochs": n_epochs,
    }
)
# Training phase
optimizer_est = optim.Adam(model_est.parameters(), lr=learning_rate)
model_est.train()
torch.manual_seed(42)
for epoch in range(n_epochs):
    for batch in estimation_loader:  # Process each batch
        batch_gpu = batch[0].to(device)  # Assuming each batch has only data, no labels

        optimizer_est.zero_grad()
        losses = {}
        num_feature_pairs = len(binary_features + continuous_features)  # Assuming binary_features includes both original and _hat features
        feature_names = binary_features + continuous_features

        for i in range(0, num_feature_pairs - 1, 2):
            feature_name = feature_names[i]
            feature_name_hat = feature_name + '_hat'

            output = model_est(batch_gpu)[:, i + 1].unsqueeze(1)
            true_values = batch_gpu[:, i].unsqueeze(1)

            if feature_name in binary_features:
                criterion = criterion_binary
            else:
                criterion = criterion_continuous

            loss = criterion(output.float(), true_values.float())
            loss.backward(retain_graph=True)
            losses[feature_name_hat] = loss.item()

        optimizer_est.step()
        optimizer_est.zero_grad()


    if epoch % 10 == 0:
        train_loss_strings = [f'{feature}: {loss:.5f}' for feature, loss in losses.items()]
        print(f'Epoch {epoch}, Training Losses: {", ".join(train_loss_strings)}')
        wandb.log({f'Training Loss - {feature}': loss for feature, loss in losses.items()}, step=epoch)

wandb.finish()

#### Y_a1
# Set all values in the 'treatment' column to 1
est_data_A1 = estimation_tensor.clone()
est_data_A1[:, treatment_column_index] = 1

probabilities_A1 = get_predicted_probabilities(model_est, est_data_A1)

#### B. get b_hat (outcome_hat) and pi_hat (treatment_hat)
outcome_hat_est = probabilities_A1['outcome_hat']
treatment_hat_est = probabilities_A1['treatment_hat']

A1 = np.ones(len(train_data))
Y_a1_train = AIPW(A1, treatment_hat_est, train_data['outcome'], outcome_hat_est)
print(f"Y_a1_train:{Y_a1_train}")

#### Y_a0
# Set all values in the 'treatment' column to 0
est_data_A0 = estimation_tensor.clone()
est_data_A0[:, treatment_column_index] = 0


#### A. get probabilities using training data
probabilities_A0 = get_predicted_probabilities(model_est, est_data_A0)

#### B. get b_hat (outcome_hat) and pi_hat (treatment_hat)
outcome_hat_est = probabilities_A0['outcome_hat']
treatment_hat_est = probabilities_A0['treatment_hat']

A0 = np.zeros(len(train_data))
Y_a0_train = AIPW(A0, treatment_hat_est, train_data['outcome'], outcome_hat_est)
print(f"Y_a0_train:{Y_a0_train}")

Phi_2 = Y_a1_train-Y_a0_train
print(f"Phi_2: {Phi_2}")
ATE = (Phi_1 + Phi_2)/2
print(f"Estimated ATE from cross-fit AIPW: {ATE}")

true_ATE = 0.034
MSE = (ATE - true_ATE) ** 2
print(f"MSE from cross-fit AIPW: {MSE}")



