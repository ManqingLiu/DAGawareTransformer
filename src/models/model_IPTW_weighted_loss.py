##### Tasks:
##### compare the ATE estimated using IPTW/MSM and g-formula in a single time point

#### steps:
#### 1. simulate the variables based on causal dag
#### for now consider:
####     4 confounders (sex, age, smoke, asthma)
####     1 treatment (qsmk)
####     1 outcome (death)
#### 2. run the model and get predictions
#### 3. calculate ATE using both IPTW
####  note that the data for g-formula will be different from the original

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

# Replace 'path_to_file.csv' with the path to your CSV file
n_samples = 5000
file_path_nointeraction = f'data/simulation{n_samples}_nointeraction.csv'
file_path_interaction= f'data/simulation{n_samples}_interaction.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path_interaction)

# Convert to PyTorch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

binary_features = torch.tensor(df[['sex', 'sex_hat', 'asthma', 'asthma_hat', 'treatment', 'treatment_hat', 'outcome', 'outcome_hat']].values, dtype=torch.long)
continuous_features = torch.tensor(df[['age_discretized', 'age_hat_discretized', 'smoke_intensity_discretized', 'smoke_intensity_hat_discretized']].values, dtype=torch.long)


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



# Create the model
num_nodes = 12  # Total number of nodes in the graph
embedding_dim = 128  # Embedding dimension
nhead = 4  # Number of heads in multihead attention
dag = [(0, 5), (0, 7), (2, 5), (2, 7), (4, 7), (8, 5), (8, 7), (10, 5), (10, 7)]  # Example DAG
categorical_dims = [2, 2, 2, 2, 2, 2, 2, 2]  # Binary features
continuous_dims = [20, 20, 20, 20]  # Continuous features, discretized into 10 bins
learning_rate = 1e-4
# categorical_dims = 2
# continuous_dims = 10

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
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion_binary = nn.BCEWithLogitsLoss()
criterion_continuous = nn.MSELoss()


binary_features = ['sex', 'sex_hat', 'asthma', 'asthma_hat', 'treatment', 'treatment_hat', 'outcome', 'outcome_hat']
continuous_features = ['age_discretized', 'age_hat_discretized', 'smoke_intensity_discretized', 'smoke_intensity_hat_discretized']
features = ['sex', 'asthma','treatment','outcome','age_discretized', 'smoke_intensity_discretized']

def validate(model, loader, criterion_binary, criterion_continuous):
    model.eval()
    val_losses = {}
    total_weighted_loss = {feature_name + '_hat': 0 for feature_name in features}
    total_weights = {feature_name + '_hat': 0 for feature_name in features}

    with torch.no_grad():
        for batch in loader:
            batch_gpu = batch[0].to(device)

            outputs = model(batch_gpu)
            num_feature_pairs = len(binary_features + continuous_features)
            feature_names = binary_features + continuous_features

            for i in range(0, num_feature_pairs - 1, 2):  # Increment by 2 to alternate between original and _hat features
                feature_name = feature_names[i]
                feature_name_hat = feature_name + '_hat'

                output = outputs[:, i + 1].unsqueeze(1)
                true_values = batch_gpu[:, i].unsqueeze(1)

                # Calculate weights for treatment_hat (assuming it's the predicted propensity score)
                if feature_name == 'treatment':
                    propensity_score = output.sigmoid()  # Convert logits to probabilities
                    treatment = true_values
                    weights = torch.where(treatment == 1, 1.0 / propensity_score, 1.0 / (1.0 - propensity_score))
                else:
                    weights = torch.ones_like(true_values)

                if feature_name in binary_features:
                    criterion = criterion_binary
                else:
                    criterion = criterion_continuous

                loss = criterion(output.float(), true_values.float())
                weighted_loss_sum = (loss * weights).sum()
                total_weighted_loss[feature_name_hat] += weighted_loss_sum.item()
                total_weights[feature_name_hat] += weights.sum().item()

    # Average the weighted losses over all batches
    for key in total_weighted_loss:
        val_losses[key] = total_weighted_loss[key] / total_weights[key] if total_weights[key] != 0 else 0

    return val_losses




import wandb
import random

n_epochs = 100
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="dag-aware-transformer-IPTW-2stage",
    name = "4 confounders (sex, age, smoke, asthma), 1 treatment (qsmk), 1 outcome (death), with interaction",

    # track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "architecture": "Transformer",
        "dataset": f"simulation with {n_samples} samples with interaction terms",
        "epochs": n_epochs,
    }
)

model.train()

for epoch in range(n_epochs):
    for batch in train_loader:  # Process each batch
        batch_gpu = batch[0].to(device)  # Assuming each batch has only data, no labels

        optimizer.zero_grad()
        losses = {}
        num_feature_pairs = len(binary_features + continuous_features)  # Assuming binary_features includes both original and _hat features
        feature_names = binary_features + continuous_features

        # Predict all outputs, including propensity scores
        outputs = model(batch_gpu)

        for i in range(0, num_feature_pairs - 1, 2):
            feature_name = feature_names[i]
            feature_name_hat = feature_name + '_hat'

            output = outputs[:, i + 1].unsqueeze(1)
            true_values = batch_gpu[:, i].unsqueeze(1)

            # Calculate weights for treatment_hat (assuming it's the predicted propensity score)
            if feature_name == 'treatment_hat':
                propensity_score = output.sigmoid()  # Convert logits to probabilities
                treatment = true_values
                weights = torch.where(treatment == 1, 1.0 / propensity_score, 1.0 / (1.0 - propensity_score))
            else:
                weights = torch.ones_like(true_values)

            if feature_name in binary_features:
                criterion = criterion_binary
            else:
                criterion = criterion_continuous

            loss = criterion(output.float(), true_values.float())
            weighted_loss_sum = (loss * weights).sum() / (weights.sum())
            weighted_loss_sum.backward(retain_graph=True)
            losses[feature_name_hat] = weighted_loss_sum.item()

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


# Usage:
probabilities = get_predicted_probabilities(model, data)

prob_A_pred = probabilities['treatment_hat']
correlation_matrix = np.corrcoef(prob_A_pred, df['treatment_probability'])

# The correlation coefficient for the two variables will be at position [0, 1] or [1, 0] in the matrix
correlation_coefficient = correlation_matrix[0, 1]

print(f'Correlation coefficient: {correlation_coefficient:.2f}')
# Convert the numpy array to a pandas DataFrame
import pandas as pd

#df = data.numpy()
#df = pd.DataFrame(df, columns=binary_features + continuous_features)
'''
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


df['prob_Y_pred_A1'] = probabilities_A1['outcome_hat']

data_A0 = data.clone()
data_A0[:, treatment_column_index] = 0
probabilities_A0 = get_predicted_probabilities(model, data_A0)

df['prob_Y_pred_A0'] = probabilities_A0['outcome_hat']
'''

df['prob_Y'] = probabilities['outcome_hat']

# Calculate mean outcome for each treatment group and calculate ATE
mean_prob_Y_pred_A1 = df[df['treatment'] == 1]['prob_Y'].mean()
mean_prob_Y_pred_A0 = df[df['treatment'] == 0]['prob_Y'].mean()


ATE = mean_prob_Y_pred_A1-mean_prob_Y_pred_A0
print(ATE)


# Calculate the true ATE
true_ATE_wxint = 0.034
true_ATE_wint = 0.0306
true_ATE = true_ATE_wint
MSE = (ATE - true_ATE) ** 2
print(MSE)

wandb.finish()