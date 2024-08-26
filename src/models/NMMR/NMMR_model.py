import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class ContinuousEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, x):
        return self.activation(self.linear(x))

class DAGTransformer(nn.Module):
    '''
    This is a transformer module that takes in the adjacency matrix of the graph
    '''
    def __init__(self,
                 dag: Dict,
                 embedding_dim: int,
                 feedforward_dim: int,
                 num_heads: int,
                 num_layers: int,
                 dropout_rate: float,
                 name: str = None):

        super(DAGTransformer, self).__init__()
        self.input_nodes = dag['input_nodes']
        self.output_nodes = dag['output_nodes']
        self.edges = dag['edges']
        self.node_ids = dag['node_ids']
        self.id2node = {v: k for k, v in self.node_ids.items()}

        self.num_nodes = len(self.node_ids.keys())
        self.embedding_dim = embedding_dim
        self.feedforward_dim = feedforward_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.name=name

        self.adj_matrix = torch.zeros(self.num_nodes, self.num_nodes)
        for source_node_name in self.edges.keys():
            source_node_id = self.node_ids[source_node_name]
            for target_node in self.edges[source_node_name]:
                target_node_id = self.node_ids[target_node]
                self.adj_matrix[source_node_id, target_node_id] = 1


        self.attn_mask = ~(self.adj_matrix.bool().T)

        # Create embeddings (input_dim is always 1 for continuous variables)
        self.embedding = nn.ModuleDict({
            node: ContinuousEmbedding(1, self.embedding_dim)
            for node in self.input_nodes.keys()
        })

        # Create encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.feedforward_dim,
            dropout=self.dropout_rate,
            activation='relu',
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Create output heads (output_dim is 1 for continuous variables)
        self.output_head = nn.ModuleDict({
            node: nn.Linear(self.embedding_dim, 1)
            for node in self.output_nodes.keys()
        })

    def forward(self, x, mask=True):
        # Convert inputs to float and calculate scaling factors
        x_float = {node: x[node].float() for node in self.input_nodes.keys()}
        input_mean = torch.stack([x_float[node].mean() for node in self.input_nodes.keys()])
        input_std = torch.stack([x_float[node].std() for node in self.input_nodes.keys()])
        input_std = torch.where(input_std == 0, torch.ones_like(input_std), input_std)  # Avoid division by zero

        # Normalize inputs
        x_normalized = {node: (x_float[node] - input_mean[i]) / input_std[i]
                        for i, node in enumerate(self.input_nodes.keys())}
        embeddings = []
        for node in self.node_ids.keys():
            # Ensure input is 2D: [batch_size, 1]
            node_input = x_normalized[node].view(-1, 1) if x_normalized[node].dim() == 1 else x_normalized[node]
            embedded = self.embedding[node](node_input)
            embeddings.append(embedded)

        x = torch.stack(embeddings, dim=1)

        if mask:
            attn_mask = self.attn_mask.repeat(x.size(0) * self.num_heads, 1, 1)
            attn_mask = attn_mask.to(x.device)
            x = self.encoder(x, mask=attn_mask)
        else:
            x = self.encoder(x)

        node_outputs = {}
        for node_name in self.output_nodes.keys():
            node_id = self.node_ids[node_name]
            node_outputs[node_name] = self.output_head[node_name](x[:, node_id, :]).squeeze(-1)

        # Denormalize only the outcome node
        outcome_node = list(self.output_nodes.keys())[0]  # Assuming there's only one output node
        if outcome_node in x_float:
            outcome_mean = x_float[outcome_node].mean()
            outcome_std = x_float[outcome_node].std()
            outcome_std = outcome_std if outcome_std != 0 else 1.0  # Avoid division by zero
            node_outputs[outcome_node] = node_outputs[outcome_node] * outcome_std + outcome_mean
        else:
            print(f"Warning: {outcome_node} not found in input data. Output will not be denormalized.")

        return node_outputs


class MLP_for_NMMR(nn.Module):
    
    def __init__(self, input_dim, train_params):
        super(MLP_for_NMMR, self).__init__()

        self.train_params = train_params
        self.network_width = train_params["network_width"]
        self.network_depth = train_params["network_depth"]

        self.layer_list = nn.ModuleList()
        for i in range(self.network_depth):
            if i == 0:
                self.layer_list.append(nn.Linear(input_dim, self.network_width))
            else:
                self.layer_list.append(nn.Linear(self.network_width, self.network_width))
        self.layer_list.append(nn.Linear(self.network_width, 1))

    def forward(self, x):
        for ix, layer in enumerate(self.layer_list):
            if ix == (self.network_depth + 1):  # if last layer, don't apply relu activation
                x = layer(x)
            else:
                x = torch.relu(layer(x))

        return x

