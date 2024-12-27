import torch.nn as nn
import torch
from typing import Dict
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import wandb
import math

from src.models.dag_aware_transformer_loss import g_formula_loss_fun, ipw_loss_fun, aipw_loss_fun
from src.dataset import CausalDataset
from src.utils import predict_function, replace_column_values
from src.train.lalonde.train_metrics import calculate_val_metrics
from src.evaluate.lalonde.evaluate_metrics import calculate_test_metrics
from src.train.acic.train_metrics import calculate_val_metrics_acic
from src.evaluate.acic.evaluate_metrics import calculate_test_metrics_acic

class DAGTransformer(nn.Module):
    def __init__(self,
                 dag: Dict,
                 network_width: int,
                 embedding_dim: int,
                 feedforward_dim: int,
                 num_heads: int,
                 num_layers: int,
                 dropout_rate: float,
                 input_layer_depth: int,
                 encoder_weight: float,
                 activation: str,
                 name: str = None):

        super(DAGTransformer, self).__init__()
        self.input_nodes = dag['input_nodes']
        self.output_nodes = dag['output_nodes']
        self.edges= dag['edges']
        self.node_ids = dag['node_ids']
        self.id2node = {v: k for k, v in self.node_ids.items()}

        self.num_nodes = len(self.node_ids.keys())
        self.network_width = network_width
        self.embedding_dim = embedding_dim
        self.feedforward_dim = feedforward_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.encoder_weight = encoder_weight
        self.activation = activation
        self.name = name

        self.adj_matrix = torch.zeros(self.num_nodes, self.num_nodes)

        for source_node_name in self.edges.keys():
            source_node_id = self.node_ids[source_node_name]
            for target_node in self.edges[source_node_name]:
                target_node_id = self.node_ids[target_node]
                self.adj_matrix[source_node_id, target_node_id] = 1

        self.attn_mask = ~(self.adj_matrix.bool().T)


        # Calculate input dimension
        self.input_dim = len(self.input_nodes)

        # Create layer list similar to MLP
        self.layer_list = nn.ModuleList()
        for i in range(input_layer_depth):
            if i == 0:
                self.layer_list.append(nn.Linear(self.num_nodes, self.network_width))
            else:
                self.layer_list.append(nn.Linear(self.network_width, self.network_width))
            self.layer_list.append(nn.ReLU())
            self.layer_list.append(nn.Dropout(self.dropout_rate))

        # Add final layer with output dimension 1
        self.layer_list.append(nn.Linear(self.network_width, 1))

        # another layer list for t
        self.layer_list_t = nn.ModuleList()
        for i in range(input_layer_depth):
            if i == 0:
                self.layer_list_t.append(nn.Linear(self.num_nodes-1, self.network_width))
            else:
                self.layer_list_t.append(nn.Linear(self.network_width, self.network_width))
            self.layer_list_t.append(nn.ReLU())
            self.layer_list_t.append(nn.Dropout(self.dropout_rate))
        # Add final layer with output dimension 1
        self.layer_list_t.append(nn.Linear(self.network_width, 1))

        # Input embedding layer
        self.input_embedding = nn.Linear(1, embedding_dim)

        # Create encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.feedforward_dim,
            dropout=self.dropout_rate,
            activation=self.activation,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Add a linear transformation for transformer_output_embeddings
        self.embed_to_scalar = nn.Linear(embedding_dim * (self.num_nodes-1), 1)
        self.embed_to_scalar_t = nn.Linear(embedding_dim * (self.num_nodes-2), 1)


    def forward(self, x, mask=False, estimator="aipw"):

        # Combine all inputs into a single tensor
        combined_input = torch.stack([x[node].float() for node in self.input_nodes.keys()], dim=1).squeeze(-1)

        # Transformer part
        # Reshape the inputs to have explicit feature dimension
        node_embeddings = []
        for node in self.input_nodes.keys():
            # Reshape input: [32] -> [32, 1]
            node_input = x[node].float().unsqueeze(-1)
            # Apply embedding: [32, 1] -> [32, embedding_dim]
            node_embedding = self.input_embedding(node_input)
            node_embeddings.append(node_embedding)

        # Stack embeddings: list of [32, embedding_dim] -> [32, num_nodes, embedding_dim]
        transformer_input = torch.stack(node_embeddings, dim=1)

        # Process the encoder
        if mask==True:
            attn_mask = self.attn_mask.repeat(transformer_input.size(0) * self.num_heads, 1, 1)
            attn_mask = attn_mask.to(x.device)
            transformer_output = self.encoder(transformer_input, mask=attn_mask)
        else:
            transformer_output = self.encoder(transformer_input)

        # Extract embeddings from transformer output
        if estimator == "aipw":
            transformer_output_t_embeddings = transformer_output[:,:-2,:].view(transformer_output.size(0), -1)
            transformer_output_t_scalar = self.embed_to_scalar_t(transformer_output_t_embeddings)
            combined_input_t = torch.cat([combined_input[:, :-2], transformer_output_t_scalar * self.encoder_weight], dim=1)
            # Process through layers
            for layer in self.layer_list_t[:-1]:  # All layers except the last
                combined_input_t = layer(combined_input_t)
            # Last layer with sigmoid to return probability
            node_output_t = torch.sigmoid(self.layer_list[-1](combined_input_t))
            transformer_output_y_embeddings = transformer_output[:,:-1,:].view(transformer_output.size(0), -1)
            transformer_output_y_scalar = self.embed_to_scalar(transformer_output_y_embeddings)
            combined_input_y = torch.cat([combined_input[:, :-1], transformer_output_y_scalar * self.encoder_weight], dim=1)
            # Process through layers
            for layer in self.layer_list[:-1]:  # All layers except the last
                combined_input_y = layer(combined_input_y)
            # Last layer without activation
            node_output_y = self.layer_list[-1](combined_input_y)
            node_output = {
                'y': node_output_y,
                't': node_output_t
            }
        elif "ipw" in estimator:
            transformer_output_t_embeddings = transformer_output[:,:-1,:].view(transformer_output.size(0), -1)
            transformer_output_t_scalar = self.embed_to_scalar(transformer_output_t_embeddings)
            combined_input_t = torch.cat([combined_input[:, :-1], transformer_output_t_scalar * self.encoder_weight], dim=1)
            # Process through layers
            for layer in self.layer_list[:-1]:
                combined_input_t = layer(combined_input_t)
            # Last layer with sigmoid to return probability
            node_output_t = torch.sigmoid(self.layer_list[-1](combined_input_t))
            node_output = {
                't': node_output_t
            }
        else:
            transformer_output_embeddings = transformer_output[:,:-1,:].view(transformer_output.size(0), -1)
            transformer_output_scalar = self.embed_to_scalar(transformer_output_embeddings)

            # Combine original input with transformer output
            combined_input = torch.cat([combined_input[:, :-1], transformer_output_scalar * self.encoder_weight], dim=1)

            # Process through layers
            for layer in self.layer_list[:-1]:  # All layers except the last
                combined_input = layer(combined_input)

            # Last layer without activation
            node_output_y = self.layer_list[-1](combined_input)

            node_output = {
                'y': node_output_y
            }

        # Split the output into individual node outputs
        node_outputs = {}
        for node_name in self.output_nodes.keys():
            node_outputs[node_name] = node_output[node_name]

        return node_outputs


    def _train(
            self,
            data_name: str,
            estimator: str,
            model: nn.Module,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            val_data: pd.DataFrame,
            pseudo_ate_data: pd.DataFrame,
            sample_id: int,
            config: Dict,
            dag: Dict,
            random_seed: int = None
    ) -> nn.Module:
        train_config = config[estimator]["training"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = model.to(device)

        opt = torch.optim.AdamW(
            model.parameters(),
            weight_decay=train_config["l2_penalty"],
            lr=train_config["learning_rate"],
        )

        wandb.init(project="DAG transformer", entity="mliu7", config=config)

        for epoch in range(train_config["n_epochs"]):
            model.train()
            for batch_ix, (batch_raw, batch_binned) in enumerate(train_dataloader):
                opt.zero_grad()
                batch = {k: v.to(device) for k, v in batch_raw.items()}
                outputs = model(batch, mask=train_config["dag_attention_mask"], estimator=estimator)

                if estimator == "g-formula":
                    y = batch_raw['y'].to(device).float()
                    y_ = torch.squeeze(outputs['y']).to(device).float()
                    batch_loss, batch_items = g_formula_loss_fun(y_, y)
                elif estimator == "ipw":
                    t = batch_raw['t'].to(device).float()
                    e = outputs['t'].to(device).squeeze().float()
                    batch_loss, batch_items = ipw_loss_fun(e, t)
                else:
                    y = batch_raw['y'].to(device).float()
                    y_ = torch.squeeze(outputs['y']).to(device).float()
                    t = batch_raw['t'].to(device)
                    e = outputs['t'].to(device).squeeze()
                    batch_loss, batch_items = aipw_loss_fun(y_, y, e, t)

                batch_loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                # print epoch number and batch loss
                #print(f"Epoch: {epoch}, Batch: {batch_ix}, Loss: {batch_loss.item()}")
                #wandb.log({f"Train: counterfactual loss": batch_loss.item()})

            model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch_raw_val, batch_binned_val in val_dataloader:
                    batch_val = {k: v.to(device) for k, v in batch_raw_val.items()}
                    outputs_val = model(batch_val, mask=train_config["dag_attention_mask"], estimator=estimator)

                    if estimator == "g-formula":
                        y = batch_val['y'].to(device).float()
                        y_ = torch.squeeze(outputs_val['y']).to(device).float()
                        val_batch_loss, val_batch_items = g_formula_loss_fun(y_, y)
                    elif estimator == "ipw":
                        t = batch_val['t'].to(device).float()
                        e = outputs_val['t'].to(device).squeeze().float()
                        val_batch_loss, val_batch_items = ipw_loss_fun(e,t)
                    else:
                        y = batch_val['y'].to(device).float()
                        y_ = torch.squeeze(outputs_val['y']).to(device).float()
                        t = batch_val['t'].to(device).float()
                        e = outputs_val['t'].to(device).squeeze().float()
                        val_batch_loss, val_batch_items = aipw_loss_fun(y_, y, e, t)

                    val_loss += val_batch_loss.item()
                    val_loss_avg = val_loss / len(val_dataloader)
            #wandb.log({f"Val: counterfactual loss": val_loss_avg})

            predictions, metrics_val = model.predict(model,
                                                           data_name,
                                                           val_data,
                                                           pseudo_ate_data,
                                                           dag=dag,
                                                           train_config=train_config,
                                                           random_seed=random_seed,
                                                            sample_id=sample_id,
                                                           prefix="Val",
                                                           estimator=estimator)

            if data_name == "lalonde_cps":
                wandb.log(metrics_val)

            if data_name == "lalonde_psid":
                wandb.log(metrics_val)

            elif data_name == "acic":
                wandb.log(metrics_val)

        return model, predictions, metrics_val


    @staticmethod
    def predict(model,
                data_name,
                data,
                pseudo_ate_data,
                dag,
                train_config: Dict,
                random_seed: int,
                sample_id: int,
                prefix: str = "Val",
                estimator: str = "ipw"):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = train_config["batch_size"]
        model = model.to(device)

        data_nodes = data[dag['nodes']]
        dataset = CausalDataset(data_nodes, dag, random_seed)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

        data_A0 = replace_column_values(data_nodes, "t", 0)
        dataset_A0 = CausalDataset(data_A0, dag, random_seed)
        dataloader_A0 = DataLoader(
            dataset_A0,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

        data_A1 = replace_column_values(data_nodes, "t", 1)
        dataset_A1 = CausalDataset(data_A1, dag, random_seed)
        dataloader_A1 = DataLoader(
            dataset_A1,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

        if estimator == "g-formula":
            predictions_y0 = predict_function(model, train_config, dataloader_A0)['y']
            # convert predictions_y0 to dataframe with column name pred_y_A0
            predictions_y0 = pd.DataFrame(predictions_y0, columns=['pred_y_A0'])
            predictions_y1 = predict_function(model, train_config, dataloader_A1)['y']
            # convert predictions_y1 to dataframe with column name pred_y_A1
            predictions_y1 = pd.DataFrame(predictions_y1, columns=['pred_y_A1'])
            predictions = pd.concat(
                [data, predictions_y0["pred_y_A0"],
                 predictions_y1["pred_y_A1"]],
                axis=1,
            )
        elif estimator=='ipw':
            predictions_t = predict_function(model, train_config, dataloader)['t']
            # convert predictions_t to dataframe with column name t_prob
            predictions_t = pd.DataFrame(predictions_t, columns=['t_prob'])
            predictions = pd.concat(
                [data, predictions_t["t_prob"]],
                axis=1,
            )
        else:
            predictions_y0 = predict_function(model, train_config, dataloader_A0)['y']
            # convert predictions_y0 to dataframe with column name pred_y_A0
            predictions_y0 = pd.DataFrame(predictions_y0, columns=['pred_y_A0'])
            predictions_y1 = predict_function(model, train_config, dataloader_A1)['y']
            # convert predictions_y1 to dataframe with column name pred_y_A1
            predictions_y1 = pd.DataFrame(predictions_y1, columns=['pred_y_A1'])
            predictions_t = predict_function(model, train_config, dataloader)['t']
            # convert predictions_t to dataframe with column name t_prob
            predictions_t = pd.DataFrame(predictions_t, columns=['t_prob'])
            predictions = pd.concat(
                [data,
                 predictions_y0["pred_y_A0"],
                 predictions_y1["pred_y_A1"],
                 predictions_t["t_prob"]],
                axis=1,
            )

        metrics = None  # Initialize metrics

        if prefix == "Val":
            if data_name == "lalonde_cps":
                metrics = calculate_val_metrics(predictions,
                                                pseudo_ate_data,
                                                sample_id,
                                                     prefix=prefix,
                                                     estimator=estimator)
            elif data_name == "lalonde_psid":
                metrics = calculate_val_metrics(predictions,
                                                pseudo_ate_data,
                                                sample_id,
                                                     prefix=prefix,
                                                     estimator=estimator)
            elif data_name == "acic":
                metrics = calculate_val_metrics_acic(predictions,
                                                          pseudo_ate_data,
                                                          prefix=prefix,
                                                          estimator=estimator,
                                                          sample_id=sample_id)

        elif prefix == "Test":
            if data_name == "lalonde_cps":
                metrics = calculate_test_metrics(predictions,
                                                      prefix=prefix,
                                                      estimator=estimator)

            elif data_name == "lalonde_psid":
                metrics= calculate_test_metrics(predictions,
                                                      prefix=prefix,
                                                      estimator=estimator)

            elif data_name == "acic":
                ps_lower_bound = train_config["ps_lower_bound"]
                ps_upper_bound = train_config["ps_upper_bound"]
                metrics = calculate_test_metrics_acic(predictions,
                                                           data['mu1'] - data['mu0'],
                                                           prefix=prefix,
                                                           estimator=estimator,
                                                           ps_lower_bound=ps_lower_bound,
                                                           ps_upper_bound=ps_upper_bound)

        return predictions, metrics

