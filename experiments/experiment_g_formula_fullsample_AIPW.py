import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.DAG_aware_transformer import TabularBERT
from src.models.utils import generate_dag_edges, ModelTrainer, rmse, IPTW_stabilized, AIPW, set_seed, seed_worker
from src.data.data_preprocess import DataProcessor
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import expit as sigmoid  # Import sigmoid function
import wandb

seed_value = 42  # This can be any number
set_seed(seed_value)

##### Part I: data pre-processing #####
#n_samples = 5000
#file_path_nointeraction = f'data/raw/simulation{n_samples}_nointeraction.csv'
#dataframe = pd.read_csv(file_path_nointeraction)
dataframe = pd.read_csv('data/realcause_datasets/lalonde_cps_sample0.csv')
# get true ATE: mean of y1 - mean of y0
ATE_true = dataframe['y1'].mean() - dataframe['y0'].mean()
print("true ATE:", ATE_true)
# remove last 3 columns of the dataframe
df = dataframe.iloc[:, :-3].copy()
#dataframe = dataframe.iloc[:, :-3]
num_bins = 15
processor = DataProcessor(df)
processor.sample_variables()
processor.bin_continuous_variables(num_bins)
tensor, feature_names = processor.create_tensor()
binary_dims, continuous_dims = processor.generate_dimensions()
binary_features, _ = processor.get_feature_names()  # Get binary and continuous feature names
dag = generate_dag_edges(feature_names)
batch_size = 32
test_size = 0.5
random_state = 1
num_workers = 20
# Split data and create DataLoaders

train_loader, train_data, val_loader, val_data, val_loader_A1, val_data_A1, val_loader_A0, val_data_A0 = (
    processor.split_data_loaders(tensor, batch_size=batch_size,test_size=test_size, random_state=random_state,
                                 feature_names=feature_names, num_workers=num_workers, seed_worker=seed_worker))


###### Part II: model training and validation ######
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
#print(torch.cuda.is_available())
#print(torch.cuda.get_device_name(0))


# Create the model
num_nodes = len(feature_names)  # Total number of nodes in the graph
embedding_dim = 128  # Embedding dimension
nhead = 4  # Number of heads in multihead attention
learning_rate = 1e-4  # Learning rate
n_epochs = 5  # Number of epochs


if wandb.run is not None:
    wandb.finish()
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="dag-aware-transformer-gformula-IPTW-AIPW",
    name = "lalonde_cps data (sample 0)",

    # track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "architecture": "Transformer",
        "dataset": f"lalonde_cps data (sample 0)",
        "epochs": n_epochs,
    }
)

model_path = f"experiments/model/model_cps_sample0_epoch{n_epochs}.pth"
# Instantiate the model, optimizer, and loss function
model = TabularBERT(num_nodes=num_nodes, embedding_dim=embedding_dim, nhead=nhead,
                    categorical_dims=binary_dims, continuous_dims=continuous_dims,
                    dag=dag, batch_size=batch_size).to(device)


model.load_state_dict(torch.load(model_path))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion_binary = nn.BCEWithLogitsLoss()
criterion_continuous = nn.MSELoss()

trainer = ModelTrainer(model, train_loader, val_loader, binary_features,
                       feature_names, criterion_binary, criterion_continuous, device)


# Train the model
for epoch in range(n_epochs):
    '''
    train_losses = trainer.train(optimizer, train_loader)
    if epoch % 1 == 0:
        train_loss_strings = [f'{feature}: {loss:.5f}' for feature, loss in train_losses.items()]
        #print(f'Epoch {epoch}, Training Losses: {", ".join(train_loss_strings)}')
        wandb.log({f'Training Loss - {feature}': loss for feature, loss in train_losses.items()}, step=epoch)

    '''
    val_losses = trainer.validate(val_loader)
    if epoch % 1 == 0:
        val_loss_strings = [f'{feature}: {loss:.5f}' for feature, loss in val_losses.items()]
        #print(f'Epoch {epoch}, Validation Losses: {", ".join(val_loss_strings)}')
        wandb.log({f'Validation Loss - {feature}': loss for feature, loss in val_losses.items()}, step=epoch)


'''
# Assuming 'model' is your trained model instance
model_path = f"experiments/model/model_cps_sample0_epoch{n_epochs}.pth"
torch.save(model.state_dict(), model_path)



#### Part IIII: Get quantities needed from predictions and derive ATE ####

model_path = f"experiments/model/model_cps_sample0_epoch{n_epochs}.pth"
model = TabularBERT(num_nodes=num_nodes, embedding_dim=embedding_dim, nhead=nhead,
                    categorical_dims=binary_dims, continuous_dims=continuous_dims,
                    dag=dag, batch_size=batch_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion_binary = nn.BCEWithLogitsLoss()
criterion_continuous = nn.MSELoss()

trainer = ModelTrainer(model, train_loader, val_loader, processor, binary_features, feature_names,
                       criterion_binary, criterion_continuous, device)


model.load_state_dict(torch.load(model_path))


#### A. G-formula and IPTW

#### A.1 G-formula

predictions_A1 = trainer.get_predictions(val_loader_A1)
val_data_A1 = val_data_A1.cpu()
val_df_A1 = pd.DataFrame(val_data_A1.numpy(), columns=feature_names)
val_df_A1['pred_y'] = processor.bin_to_original(predictions_A1['y_hat_bin'], 'y_hat')
val_df_A1['pred_t'] = sigmoid(predictions_A1['t_hat'])
#print(val_df_A1[['pred_y','pred_t']].describe())

# get means of pred_y
mean_pred_y_A1 = val_df_A1['pred_y'].mean()

# repeat from above for df_A0
predictions_A0 = trainer.get_predictions(val_loader_A0)
val_data_A0 = val_data_A0.cpu()
val_df_A0 = pd.DataFrame(val_data_A0.numpy(), columns=feature_names)
val_df_A0['pred_y'] = processor.bin_to_original(predictions_A0['y_hat_bin'], 'y_hat')
val_df_A0['pred_t'] = sigmoid(predictions_A0['t_hat'])
#print(val_df_A0[['pred_y','pred_t']].describe())

# get means of pred_y
mean_pred_y_A0 = val_df_A0['pred_y'].mean()

ATE_standardization = mean_pred_y_A1-mean_pred_y_A0
print("predicted ATE from g-formula:", ATE_standardization)
rmse_standardization = rmse(ATE_standardization, ATE_true)
print("RMSE from standardization:", rmse_standardization)



#### A.2 IPTW
predictions = trainer.get_predictions(val_loader)

# Now, feature_names contains all the keys/feature names from predictions_dict
# get the predicted feature names
#pred_feature_names = list(predictions.keys())
#print("Predicted Feature Names:", pred_feature_names)
val_data = val_data.cpu()
val_df = pd.DataFrame(val_data.numpy(), columns=feature_names)
val_df['pred_y'] = processor.bin_to_original(predictions['y_hat_bin'], 'y_hat')
val_df['pred_t'] = sigmoid(predictions['t_hat'])


train_dataset, val_dataset = train_test_split(df, test_size=test_size, random_state=random_state)
val_df['y'] = val_dataset['y']
#print(val_df[['pred_y','pred_t','y']].describe())

ATE_IPTW = IPTW_stabilized(val_df['t'],  val_df['y'], val_df['pred_t'])
print("predicted ATE from stabilized IPTW:", ATE_IPTW)

rmse_IPTW = rmse(ATE_IPTW, ATE_true)
print("RMSE from stabilized IPTW:", rmse_IPTW)


#### B. Naive full-sample AIPW estimator
Y_a1 = AIPW(val_df['t'], val_df_A1['pred_t'], val_df['y'], val_df_A1['pred_y'])
Y_a0 = AIPW(val_df['t'], val_df_A0['pred_t'], val_df['y'], val_df_A0['pred_y'])

ATE_AIPW = Y_a1-Y_a0
print(f"Estimated ATE from AIPW (DR): {ATE_AIPW}")

rmse_AIPW = rmse(ATE_AIPW, ATE_true)
print("RMSE from AIPW (DR):", rmse_AIPW)
'''
wandb.finish()
