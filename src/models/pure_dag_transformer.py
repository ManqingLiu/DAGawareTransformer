import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class CustomTransformerEncoderLayer(nn.Module):
    """Custom transformer encoder layer without layer normalization"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                               batch_first=batch_first)

        # Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Self-attention block
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask,
                              is_causal=is_causal)[0]
        src = src + self.dropout1(src2)

        # Feedforward block
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)

        return src


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
                 use_layernorm: bool = False,
                 name: str = None):

        super(DAGTransformer, self).__init__()
        self.input_nodes = dag['input_nodes']
        self.output_nodes = dag['output_nodes']
        self.edges = dag['edges']
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
        self.use_layernorm = use_layernorm
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
                self.layer_list_t.append(nn.Linear(self.num_nodes - 1, self.network_width))
            else:
                self.layer_list_t.append(nn.Linear(self.network_width, self.network_width))
            self.layer_list_t.append(nn.ReLU())
            self.layer_list_t.append(nn.Dropout(self.dropout_rate))
        # Add final layer with output dimension 1
        self.layer_list_t.append(nn.Linear(self.network_width, 1))

        # Input embedding layer
        self.input_embedding = nn.Linear(1, embedding_dim)

        # Create encoder layers - use custom layer if layernorm is disabled
        if self.use_layernorm:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=self.num_heads,
                dim_feedforward=self.feedforward_dim,
                dropout=self.dropout_rate,
                activation=self.activation,
                batch_first=True
            )
        else:
            encoder_layer = CustomTransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=self.num_heads,
                dim_feedforward=self.feedforward_dim,
                dropout=self.dropout_rate,
                activation=self.activation,
                batch_first=True
            )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Add a linear transformation for transformer_output_embeddings
        self.embed_to_scalar = nn.Linear(embedding_dim * (self.num_nodes - 1), 1)
        self.embed_to_scalar_t = nn.Linear(embedding_dim * (self.num_nodes - 2), 1)

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
        if mask == True:
            attn_mask = self.attn_mask.repeat(transformer_input.size(0) * self.num_heads, 1, 1)
            attn_mask = attn_mask.to(x[list(x.keys())[0]].device)
            transformer_output = self.encoder(transformer_input, mask=attn_mask)
        else:
            transformer_output = self.encoder(transformer_input)

        # Extract embeddings from transformer output
        if estimator == "aipw":
            transformer_output_t_embeddings = transformer_output[:, :-2, :].view(transformer_output.size(0), -1)
            transformer_output_t_scalar = self.embed_to_scalar_t(transformer_output_t_embeddings)
            combined_input_t = torch.cat([combined_input[:, :-2], transformer_output_t_scalar * self.encoder_weight],
                                         dim=1)
            # Process through layers
            for layer in self.layer_list_t[:-1]:  # All layers except the last
                combined_input_t = layer(combined_input_t)
            # Last layer with sigmoid to return probability
            node_output_t = torch.sigmoid(self.layer_list_t[-1](combined_input_t))

            transformer_output_y_embeddings = transformer_output[:, :-1, :].view(transformer_output.size(0), -1)
            transformer_output_y_scalar = self.embed_to_scalar(transformer_output_y_embeddings)
            combined_input_y = torch.cat([combined_input[:, :-1], transformer_output_y_scalar * self.encoder_weight],
                                         dim=1)
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
            transformer_output_t_embeddings = transformer_output[:, :-1, :].view(transformer_output.size(0), -1)
            transformer_output_t_scalar = self.embed_to_scalar(transformer_output_t_embeddings)
            combined_input_t = torch.cat([combined_input[:, :-1], transformer_output_t_scalar * self.encoder_weight],
                                         dim=1)
            # Process through layers
            for layer in self.layer_list[:-1]:
                combined_input_t = layer(combined_input_t)
            # Last layer with sigmoid to return probability
            node_output_t = torch.sigmoid(self.layer_list[-1](combined_input_t))
            node_output = {
                't': node_output_t
            }

        else:
            transformer_output_embeddings = transformer_output[:, :-1, :].view(transformer_output.size(0), -1)
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