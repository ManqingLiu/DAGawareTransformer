import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.optim as optim
from src.models.DAG_aware_transformer_gate import TabularBERT
from src.models.utils import *
from src.data.data_preprocess import DataProcessor
from sklearn.model_selection import train_test_split
from config import *
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import expit as sigmoid  # Import sigmoid function
import wandb

set_seed()

##### Part I: data pre-processing #####
dataset_type = 'lalonde_psid'  # Can be 'cps' or 'psid', depending on what you want to use

# Use the variable in the file path
dataframe = pd.read_csv(f'data/realcause_datasets/{dataset_type}_sample{N_SAMPLE}.csv')


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
fixed_edges, dynamic_edge_indices = generate_dag_edges(feature_names)
#print(feature_names)
#print(dag)
# Split data and create DataLoaders


train_loader, train_data, val_loader, val_data, \
val_loader_A1, val_data_A1, val_loader_A0, val_data_A0, \
train_loader_A1, train_data_A1, train_loader_A0, train_data_A0 = (
    processor.split_data_loaders(tensor, batch_size=BATCH_SIZE, test_size=TEST_SIZE, random_state=SEED_VALUE,
                                 feature_names=feature_names))



###### Part II: model training and validation ######
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))


# Create the model
num_nodes = len(feature_names)  # Total number of nodes in the graph


# Instantiate the model, optimizer, and loss function
model = TabularBERT(num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM, nhead=N_HEAD,
                    dag_edges=fixed_edges, dynamic_edge_indices=dynamic_edge_indices,
                    batch_size=BATCH_SIZE,
                    categorical_dims=binary_dims, continuous_dims=continuous_dims,
                    device=device, dropout_rate=DROPOUT_RATE, all_features=feature_names)

# model.load_state_dict(torch.load(model_path))
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion_binary = nn.BCEWithLogitsLoss()
criterion_continuous = nn.MSELoss()


trainer = ModelTrainer(model, train_loader, val_loader, binary_features,
                       feature_names, criterion_binary, criterion_continuous, device)

train_dataset, val_dataset = train_test_split(df, test_size=TEST_SIZE, random_state=SEED_VALUE)


#### Part I: G-formula ####
# Correct model_type usage in filenames and ensure correct loader/data tensor usage
conditions = {
    "A1": {"train": (val_loader_A1, val_data_A1), "val": (train_loader_A1, train_data_A1)},
    "A0": {"train": (val_loader_A0, val_data_A0), "val": (train_loader_A0, train_data_A0)}
}

# Initialize empty dictionaries to store DataFrames for A1 and A0 conditions
dfs_standardization = {"A1": [], "A0": []}

mean_pred_y = {}

for condition, model_types in conditions.items():
    dfs = []

    for model_type, (loader, data) in model_types.items():
        # Assume correct model_path is constructed
        model_path = (f"experiments/model/{dataset_type}/model_{dataset_type}_{model_type}_sample{N_SAMPLE}_"
                      f"epoch{N_EPOCHS}_lr{LEARNING_RATE:.4f}_wd{WEIGHT_DECAY:.4f}_dr{DROPOUT_RATE:.4f}_bs{BATCH_SIZE}_ed{EMBEDDING_DIM}_nh{N_HEAD}.pth")

        # Ensure correct data loading based on model type
        df = predict_and_create_df(model, model_path, device, trainer, loader, data, processor, feature_names, dataset_type)

        # Add a column to indicate the source of the data
        df['data_source'] = model_type  # Adds "train" or "val" to each row

        dfs.append(df)

    # Now, instead of creating one df_standardization, you create separate ones for each condition
    dfs_standardization[condition] = pd.concat(dfs)
    mean_pred_y[condition] = dfs_standardization[condition]['pred_y'].mean()

# At this point, dfs_standardization['A1'] and dfs_standardization['A0'] are your two DataFrames
# You can access them directly for further processing or analysis
df_standardization_A1 = dfs_standardization['A1']
df_standardization_A0 = dfs_standardization['A0']

ATE_estimated = mean_pred_y["A1"] - mean_pred_y["A0"]
print(f"Predicted ATE from standardization: {ATE_estimated:.4f}")

# Calculate RMSE assuming you have defined the rmse function and have the true ATE value
rmse_standardization = rmse(ATE_estimated, ATE_true)
print(f"RMSE from standardization: {rmse_standardization:.4f}")



#### A.2 IPTW
del df
# Loaders and corresponding data tensors for both training and validation sets

scenarios = {
    "train": (val_loader, val_data, val_dataset),  # Model trained on training data predicting on validation data
    "val": (train_loader, train_data, train_dataset)  # Model trained on validation data predicting on training data
}

# Combine predictions from both models
df_IPTW = pd.DataFrame()

for model_type, (loader, data, dataset) in scenarios.items():
    #model.load_state_dict(torch.load(model_path))
    model_path = (f"experiments/model/{dataset_type}/model_{dataset_type}_{model_type}_sample{N_SAMPLE}_"
                  f"epoch{N_EPOCHS}_lr{LEARNING_RATE:.4f}_wd{WEIGHT_DECAY:.4f}_dr{DROPOUT_RATE:.4f}_bs{BATCH_SIZE}_ed{EMBEDDING_DIM}_nh{N_HEAD}.pth")

    df = predict_and_create_df(model, model_path, device, trainer, loader, data, processor, feature_names, dataset_type)
    df['y'] = dataset['y']
    df['data_source'] = model_type
    df_IPTW = pd.concat([df_IPTW, df])

# Assuming 't' and 'y' columns exist in the combined_df or are added from the original dataset
ATE_IPTW = IPTW_stabilized(df_IPTW['t'], df_IPTW['y'], df_IPTW['pred_t'])
print(f"Predicted ATE from IPTW: {ATE_IPTW:.4f}")
rmse_IPTW = rmse(ATE_IPTW, ATE_true)
print(f"RMSE from IPTW: {rmse_IPTW:.4f}")


#### AIPW estimator
val_df_A1 = df_standardization_A1.loc[df_standardization_A1['data_source'] == 'val']
train_df_A1 = df_standardization_A1.loc[df_standardization_A1['data_source'] == 'train']
val_df = df_IPTW.loc[df_IPTW['data_source'] == 'val']
train_df = df_IPTW.loc[df_IPTW['data_source'] == 'train']
Y_a1_val = AIPW(val_df['t'], val_df_A1['pred_t'], val_df['y'], val_df_A1['pred_y'])
Y_a1_train = AIPW(train_df['t'], train_df_A1['pred_t'], train_df['y'], train_df_A1['pred_y'])

Y_a1 = (Y_a1_val + Y_a1_train)/2


# repeat for a0
val_df_A0 = df_standardization_A0.loc[df_standardization_A0['data_source'] == 'val']
train_df_A0 = df_standardization_A0.loc[df_standardization_A0['data_source'] == 'train']
Y_a0_val = AIPW(val_df['t'], val_df_A0['pred_t'], val_df['y'], val_df_A0['pred_y'])
Y_a0_train = AIPW(train_df['t'], train_df_A0['pred_t'], train_df['y'], train_df_A0['pred_y'])

Y_a0 = (Y_a0_val + Y_a0_train)/2

ATE_AIPW = Y_a1-Y_a0
print(f"Estimated ATE from AIPW (DR): {ATE_AIPW:.4f}")

rmse_AIPW = rmse(ATE_AIPW, ATE_true)
print(f"RMSE from AIPW (DR): {rmse_AIPW:.4f}")
