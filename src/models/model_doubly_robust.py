##### Tasks:
##### compare the ATE estimated using IPTW/MSM and g-formula in a single time point

#### steps:
#### 1. simulate the variables based on causal dag
#### for now consider:
####     4 confounders (sex, age, smoke, asthma)
####     1 treatment (qsmk)
####     1 outcome (death)
#### 2. run the model and get predictions
#### 3. calculate ATE using g-formula
####  note that the data for g-formula will be different from the original

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
#from model_IPTW import TabularBERT, validate, get_predicted_probabilities

class TabularBERT(nn.Module):
    def __init__(self, num_nodes, embedding_dim, nhead, dag, categorical_dims, continuous_dims):
        super(TabularBERT, self).__init__()

        # Total number of features
        # total_features = len(categorical_dims) + len(continuous_dims)

        # Embeddings for all features
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embedding_dim) for dim in categorical_dims + continuous_dims
        ])
        # print(self.embeddings)

        self.attention = nn.MultiheadAttention(embedding_dim, nhead)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1)

        self.adj_matrix = torch.zeros(num_nodes, num_nodes)
        for edge in dag:
            self.adj_matrix[edge[0], edge[1]] = 1
        self.adj_mask = (1 - self.adj_matrix) * -1e9
        self.adj_mask = self.adj_mask.to(device)

        self.output_layers = nn.ModuleList([nn.Linear(embedding_dim, 1) for _ in range(num_nodes)])
        self.nhead = nhead

    def forward(self, x):
        # Split the tensor into individual features and apply corresponding embeddings

        embeddings = [self.embeddings[i](x[:, i]) for i in range(x.size(1))]
        # Assuming all tensors in 'embeddings' have shape [batch_size, embedding_dim]
        x = torch.stack(embeddings)

        # 'concatenated_embeddings' now has a shape of [seq_len, batch_size, embedding_dim]
        # print(x.size())
        # x = torch.cat(embeddings, dim=1)
        # print(x.size())
        # x = x.permute(1, 0, 2)  # Transformer expects seq_len, batch_size, input_dim

        batch_size = x.size(1)
        attn_mask = self.adj_mask.repeat(batch_size * self.nhead, 1, 1)

        attn_output, _ = self.attention(x, x, x, attn_mask=attn_mask)
        encoder_output = self.transformer_encoder(attn_output)
        decoder_output = self.transformer_decoder(encoder_output, encoder_output)
        decoder_output = decoder_output.permute(1, 0, 2)

        outputs = [output_layer(decoder_output[:, i, :]) for i, output_layer in enumerate(self.output_layers)]
        return torch.cat(outputs, dim=1)

def validate(model, loader, criterion_binary, criterion_continuous):
    model.eval()
    val_losses = {}
    total_loss_count = {feature_name + '_hat': 0 for feature_name in features}

    with torch.no_grad():
        for batch in loader:
            batch_gpu = batch[0].to(device)

            num_feature_pairs = len(binary_features + continuous_features)
            feature_names = binary_features + continuous_features

            for i in range(0, num_feature_pairs - 1, 2):  # Increment by 2 to alternate between original and _hat features
                feature_name = feature_names[i]
                feature_name_hat = feature_name + '_hat'

                output = model(batch_gpu)[:, i + 1].unsqueeze(1)  # Predictions for _hat features
                true_values = batch_gpu[:, i].unsqueeze(1)  # Original features

                # Determine if the current feature is binary or continuous
                if feature_name in binary_features:
                    criterion = criterion_binary
                else:
                    criterion = criterion_continuous

                loss = criterion(output.float(), true_values.float())
                total_loss_count[feature_name_hat] += loss.item()

    # Average the losses over all batches
    num_batches = len(loader)
    for key in total_loss_count:
        val_losses[key] = total_loss_count[key] / num_batches

    return val_losses


def get_predicted_probabilities(model, data):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        probabilities = torch.sigmoid(model(data))
    # Assuming binary_features includes both original and _hat features
    num_feature_pairs = len(binary_features+continuous_features)
    feature_names = binary_features + continuous_features
    prob_dict = {feature_names[i]: probabilities[:, i].cpu().numpy() for i in range(1, num_feature_pairs-2, 2)}
    return prob_dict

n_samples = 5000
file_path_nointeraction = f'data/simulation{n_samples}_nointeraction.csv'
file_path_interaction = f'data/simulation{n_samples}_interaction.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path_nointeraction)

'''
# Randomly shuffle the DataFrame and then split it into two halves
df_randomized = df.sample(frac=1, random_state=42)  # Shuffle the DataFrame
split_point = len(df_randomized) // 2  # Find the splitting point, which is half the length of the DataFrame

# Split the DataFrame into two halves
df_training = df_randomized[:split_point]
df_estimation = df_randomized[split_point:]
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert to PyTorch tensors
binary_features = torch.tensor(df[['sex', 'sex_hat', 'asthma', 'asthma_hat', 'treatment', 'treatment_hat', 'outcome', 'outcome_hat']].values, dtype=torch.long)
continuous_features = torch.tensor(df[['age_discretized', 'age_hat_discretized', 'smoke_intensity_discretized', 'smoke_intensity_hat_discretized']].values, dtype=torch.long)

# Create the model
num_nodes = 12  # Total number of nodes in the graph
embedding_dim = 128  # Embedding dimension
nhead = 4  # Number of heads in multihead attention
dag = [(0, 5), (0, 7), (2, 5), (2, 7), (4, 7), (8, 5), (8, 7), (10, 5), (10, 7)]  # Example DAG
categorical_dims = [2, 2, 2, 2, 2, 2, 2, 2]  # Binary features
continuous_dims = [20, 20, 20, 20]  # Continuous features, discretized into 10 bins
# categorical_dims = 2
# continuous_dims = 10
model = TabularBERT(num_nodes, embedding_dim, nhead, dag, categorical_dims, continuous_dims)

# Concatenate binary and continuous features
data = torch.cat([binary_features, continuous_features], dim=1)

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# Split the data into training, validation, and testing sets
train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Create DataLoader objects for the training, validation, and testing sets
train_loader = DataLoader(TensorDataset(train_data), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(val_data), batch_size=32)
test_loader = DataLoader(TensorDataset(test_data), batch_size=32)

# Instantiate the model, optimizer, and loss function
model = TabularBERT(num_nodes=num_nodes, embedding_dim=128, nhead=4, dag=dag,
                    categorical_dims=categorical_dims, continuous_dims=continuous_dims)
model = model.to(device)
learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion_binary = nn.BCEWithLogitsLoss()
criterion_continuous = nn.MSELoss()


import wandb
import random

n_epochs = 100
if wandb.run is not None:
    wandb.finish()
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="dag-aware-transformer-doubly-robust",
    name = "4 confounders (sex, age, smoke, asthma), 1 treatment (qsmk), 1 outcome (death)",

    # track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "architecture": "Transformer",
        "dataset": f"simulation with {n_samples} samples without interaction terms",
        "epochs": n_epochs,
    }
)

binary_features = ['sex', 'sex_hat', 'asthma', 'asthma_hat', 'treatment', 'treatment_hat', 'outcome', 'outcome_hat']
continuous_features = ['age_discretized', 'age_hat_discretized', 'smoke_intensity_discretized', 'smoke_intensity_hat_discretized']
features = ['sex', 'asthma','treatment','outcome','age_discretized', 'smoke_intensity_discretized']


model.train()

for epoch in range(n_epochs):
    for batch in train_loader:  # Process each batch
        batch_gpu = batch[0].to(device)  # Assuming each batch has only data, no labels

        optimizer.zero_grad()
        losses = {}
        num_feature_pairs = len(binary_features + continuous_features)  # Assuming binary_features includes both original and _hat features
        feature_names = binary_features + continuous_features

        for i in range(0, num_feature_pairs - 1, 2):
            feature_name = feature_names[i]
            feature_name_hat = feature_name + '_hat'

            output = model(batch_gpu)[:, i + 1].unsqueeze(1)
            true_values = batch_gpu[:, i].unsqueeze(1)

            if feature_name in binary_features:
                criterion = criterion_binary
            else:
                criterion = criterion_continuous

            loss = criterion(output.float(), true_values.float())
            loss.backward(retain_graph=True)
            losses[feature_name_hat] = loss.item()

        optimizer.step()
        optimizer.zero_grad()


    if epoch % 10 == 0:
        train_loss_strings = [f'{feature}: {loss:.5f}' for feature, loss in losses.items()]
        print(f'Epoch {epoch}, Training Losses: {", ".join(train_loss_strings)}')
        wandb.log({f'Training Loss - {feature}': loss for feature, loss in losses.items()}, step=epoch)

    # Validation phase
    val_losses = validate(model, val_loader, criterion_binary, criterion_continuous)
    if epoch % 10 == 0:
        val_loss_strings = [f'{feature}: {loss:.5f}' for feature, loss in val_losses.items()]
        print(f'Epoch {epoch}, Validation Losses: {", ".join(val_loss_strings)}')
        wandb.log({f'Validation Loss - {feature}': loss for feature, loss in val_losses.items()}, step=epoch)

# After training, evaluate on the test set
test_losses = validate(model, test_loader, criterion_binary, criterion_continuous)

# Optional: Print test losses
test_loss_strings = [f'{feature}: {loss:.5f}' for feature, loss in test_losses.items()]
print(f'Test Losses: {", ".join(test_loss_strings)}')


feature_to_index = {'sex': 0, 'sex_hat': 1, 'asthma': 2, 'asthma_hat': 3, 'treatment': 4, 'treatment_hat': 5,
                    'outcome': 6, 'outcome_hat': 7,
                    'age_discretized': 8, 'age_hat_discretized': 9, 'smoke_intensity_discretized': 10,
                    'smoke_intensity_hat_discretized': 11}

# Now you can use the dictionary to reference the 'treatment' column
treatment_column_index = feature_to_index['treatment']

# Set all values in the 'treatment' column to 1
data_A1 = data.clone()
data_A1[:, treatment_column_index] = 1
probabilities_A1 = get_predicted_probabilities(model, data_A1)


outcome_hat_A1 = probabilities_A1['outcome_hat']
treatment_hat_A1 = probabilities_A1['treatment_hat']

data_A0 = data.clone()
data_A0[:, treatment_column_index] = 0
probabilities_A0 = get_predicted_probabilities(model, data_A0)

outcome_hat_A0 = probabilities_A0['outcome_hat']
treatment_hat_A0 = probabilities_A0['treatment_hat']


def AIPW(treatment, treatment_hat, outcome, outcome_hat):
    """
    Calculates the full sample AIPW estimator of E[Y^a].

    Parameters:
    - treatment: np.array, actual treatment received.
    - treatment_hat: np.array, predicted probability of receiving the treatment.
    - outcome: np.array, actual outcome observed.
    - outcome_hat: np.array, predicted outcome.

    Returns:
    - float, E[Y^a].
    """

    if len(treatment) != len(treatment_hat) or len(outcome) != len(outcome_hat) or len(treatment) != len(outcome):
        raise ValueError("All input arrays must have the same length.")

    Y_a = outcome_hat + (treatment / treatment_hat) * (outcome - outcome_hat)
    return np.mean(Y_a)


A1 = np.ones(n_samples)
A0 = np.zeros(n_samples)
Y_a1 = AIPW(A1, treatment_hat_A1, df['outcome'], outcome_hat_A1)
Y_a0 = AIPW(A0, treatment_hat_A0, df['outcome'], outcome_hat_A0)


# Example usage:
# treatment = np.array([1, 0, 1, 0, 1])  # Actual treatment received
# treatment_hat = np.array([0.8, 0.1, 0.9, 0.2, 0.7])  # Predicted probability of receiving the treatment
# outcome = np.array([5, 1, 6, 2, 7])  # Actual outcomes observed
# outcome_hat = np.array([4, 2, 5, 3, 6])  # Predicted outcomes

# print(adjusted_outcome_mean(treatment, treatment_hat, outcome, outcome_hat))


ATE = Y_a1-Y_a0
print(f"Estimated ATE: {ATE}")



# Calculate the true ATE
true_ATE = 0.034
MSE = (ATE - true_ATE) ** 2
print(f"MSE: {MSE}")

'''
nconfounder = 4
df_treat_1.to_excel(f"data/dataL{nconfounder}+{n_samples}-g-formula-interaction-A1.xlsx", index=False)
df_treat_0.to_excel(f"data/dataL{nconfounder}+{n_samples}-g-formula-interaction-A0.xlsx", index=False)
'''

wandb.finish()