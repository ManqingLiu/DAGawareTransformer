import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from src.models.DAG_aware_transformer import TabularBERT
from src.models.utils import *
from src.data.data_preprocess import DataProcessor
from config_cps import *


set_seed()

# Wrap your training code in a new function
def train_and_save_model(sample_number, dataset_type):
    #dataset_type = 'lalonde_cps'  # Can be 'cps' or 'psid'

    # Use the variable in the file path
    dataframe = pd.read_csv(f'data/realcause_datasets/{dataset_type}_sample{sample_number}.csv')

    df = dataframe.iloc[:, :-3].copy()
    num_bins = NUM_BINS
    processor = DataProcessor(df)
    processor.sample_variables()
    processor.bin_continuous_variables(num_bins)
    tensor, feature_names = processor.create_tensor()
    binary_dims, continuous_dims = processor.generate_dimensions()
    binary_features, _ = processor.get_feature_names()  # Get binary and continuous feature names
    dag = generate_dag_edges(feature_names)


    train_loader, train_data, val_loader, val_data, \
    val_loader_A1, val_data_A1, val_loader_A0, val_data_A0, \
    train_loader_A1, train_data_A1, train_loader_A0, train_data_A0 = (
        processor.split_data_loaders(tensor, batch_size=BATCH_SIZE, test_size=TEST_SIZE, random_state=SEED_VALUE,
                                     feature_names=feature_names))



    ###### Part II: model training and validation ######
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the model
    num_nodes = len(feature_names)  # Total number of nodes in the graph


    # Instantiate the model, optimizer, and loss function
    model = TabularBERT(num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM, nhead=N_HEAD,
                        categorical_dims=binary_dims, continuous_dims=continuous_dims,
                        dag=dag, batch_size=BATCH_SIZE, device=device, dropout_rate=DROPOUT_RATE)

    # model.load_state_dict(torch.load(model_path))
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion_binary = nn.BCEWithLogitsLoss()
    criterion_continuous = nn.MSELoss()


    trainer = ModelTrainer(model, train_loader, val_loader, binary_features,
                           feature_names, criterion_binary, criterion_continuous, device)

    # Train the model
    loaders = {
        "train": train_loader,
        "val": val_loader
    }


    for loader_name, loader in loaders.items():

        for epoch in range(N_EPOCHS):
            # Train on the current loader
            trainer.train(optimizer, train_loader)
            trainer.validate(val_loader)

        # Save the model after training on each loader
        save_model(model, dataset_type, loader_name, sample_number, N_EPOCHS,
                   LEARNING_RATE, WEIGHT_DECAY, DROPOUT_RATE, BATCH_SIZE, EMBEDDING_DIM, N_HEAD)
        print(f"Model saved after training on {loader_name} data")


for sample_number in range(50, 100):
    train_and_save_model(sample_number, 'lalonde_cps')