import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.DAG_aware_transformer import TabularBERT
from src.models.utils import generate_dag_edges, ModelTrainer, rmse, IPTW_stabilized, AIPW
from src.dataset.data_preprocess import DataProcessor
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import expit as sigmoid  # Import sigmoid function
from sklearn.metrics import f1_score
import wandb

file_path = f'data/unmeasured_confounding/df.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

##### Part I: data pre-processing #####
#n_samples = 5000
#file_path_nointeraction = f'data/raw/simulation{n_samples}_nointeraction.csv'
#dataframe = pd.read_csv(file_path_nointeraction)
# get true ATE: mean of y1 - mean of y0
ATE_true = 0.5
print("true ATE:", ATE_true)
num_bins = 15
processor = DataProcessor(df)
np.random.seed(0)
processor.sample_variables()
processor.bin_continuous_variables(num_bins)
tensor, feature_names = processor.create_tensor()
binary_dims, continuous_dims = processor.generate_dimensions()
binary_features, _ = processor.get_feature_names()  # Get binary and continuous feature names
dag = generate_dag_edges(feature_names)
print(feature_names)
print(dag)


batch_size = 32
test_size = 0.3
random_state = 1
# Split data and create DataLoaders

train_loader, train_data, val_loader, val_data, val_loader_A1, val_data_A1, val_loader_A0, val_data_A0 = (
    processor.split_data_loaders(tensor, batch_size=batch_size,test_size=test_size, random_state=random_state, feature_names=feature_names))

###### Part II: model training and validation ######
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the model
num_nodes = len(feature_names)  # Total number of nodes in the graph
embedding_dim = 128  # Embedding dimension
nhead = 4  # Number of heads in multihead attention
learning_rate = 1e-4  # Learning rate
n_epochs = 20  # Number of epochs


if wandb.run is not None:
    wandb.finish()
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="dag-aware-transformer-unmeasured-confounding",
    name = "data (sample 0 - no u_hat)",

    # track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "architecture": "Transformer",
        "dataset": f"data (sample 0 - no u_hat)",
        "epochs": n_epochs,
    }
)

# Instantiate the model, optimizer, and loss function
model = TabularBERT(num_nodes=num_nodes, embedding_dim=embedding_dim, nhead=nhead,
                    categorical_dims=binary_dims, continuous_dims=continuous_dims,
                    dag=dag, batch_size=batch_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion_binary = nn.BCEWithLogitsLoss()
criterion_continuous = nn.MSELoss()

trainer = ModelTrainer(model, train_loader, val_loader, binary_features,
                       feature_names, criterion_binary, criterion_continuous, device)

# Train the model
for epoch in range(n_epochs):
    train_losses = trainer.train(optimizer, train_loader)
    if epoch % 1 == 0:
        train_loss_strings = [f'{feature}: {loss:.5f}' for feature, loss in train_losses.items()]
        #print(f'Epoch {epoch}, Training Losses: {", ".join(train_loss_strings)}')
        wandb.log({f'Training Loss - {feature}': loss for feature, loss in train_losses.items()}, step=epoch)

    val_losses = trainer.validate(val_loader)
    if epoch % 1 == 0:
        val_loss_strings = [f'{feature}: {loss:.5f}' for feature, loss in val_losses.items()]
        #print(f'Epoch {epoch}, Validation Losses: {", ".join(val_loss_strings)}')
        wandb.log({f'Validation Loss - {feature}': loss for feature, loss in val_losses.items()}, step=epoch)



# Assuming 'model' is your trained model instance
# model_path = f"experiments/model/model_u_sample0_epoch{n_epochs}.pth"
# torch.save(model.state_dict(), model_path)

predictions = trainer.get_predictions(val_loader)


# Get the validation data and predictions of u
val_data = val_data.cpu()
val_df = pd.DataFrame(val_data.numpy(), columns=feature_names)
val_df['pred_y'] = processor.bin_to_original(predictions['y_hat_bin'], 'y_hat')
val_df['pred_t'] = sigmoid(predictions['t_hat'])


train_dataset, val_dataset = train_test_split(df, test_size=test_size, random_state=random_state)
val_df['y'] = val_dataset['y']
# print(val_df[['pred_y','pred_t','y']].describe())

# accuracy of the predicted treatment
val_df['pred_t_bin'] = np.where(val_df['pred_t'] > 0.5, 1, 0)
accuracy_t = f1_score(val_df['t'], val_df['pred_t_bin'])
print("accuracy of the predicted t (F-1 score):", accuracy_t)


ATE_IPTW = IPTW_stabilized(val_df['t'],  val_df['y'], val_df['pred_t'])
print("predicted ATE from stabilized IPTW:", ATE_IPTW)

rmse_IPTW = rmse(ATE_IPTW, ATE_true)
print("RMSE from stabilized IPTW:", rmse_IPTW)
