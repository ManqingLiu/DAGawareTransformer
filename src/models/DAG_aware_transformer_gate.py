##### DAG aware transformer - Model architecture
import torch
import torch.nn as nn
import math


class TabularBERT(nn.Module):
    def __init__(self, num_nodes, embedding_dim, nhead, dag_edges, dynamic_edge_indices, batch_size, categorical_dims,
                 continuous_dims, device, dropout_rate, all_features):
        super(TabularBERT, self).__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.all_features = all_features

        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embedding_dim) for dim in categorical_dims + continuous_dims
        ])

        self.attention = nn.MultiheadAttention(embedding_dim, nhead, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        # Fixed adjacency matrix
        self.fixed_adj_matrix = torch.zeros(num_nodes, num_nodes)
        for edge in dag_edges:
            self.fixed_adj_matrix[edge[0], edge[1]] = 1

        # Learnable adjacency matrix for dynamic edges
        self.dynamic_adj_matrix = nn.Parameter(torch.zeros(num_nodes, num_nodes))

        self.output_layers = nn.ModuleList([nn.Linear(embedding_dim, 1) for _ in range(num_nodes)])
        self.nhead = nhead
        self.batch_size = batch_size
        self.dynamic_edge_indices = dynamic_edge_indices

        # Create the adjacency mask
        self.adj_mask = (1 - self.fixed_adj_matrix) * -1e9
        self.adj_mask = self.adj_mask.to(device)

    def forward(self, x):
        embeddings = [self.embeddings[i](x[:, i]) for i in range(x.size(1))]
        x = torch.stack(embeddings)
        x = x.permute(1, 0, 2)

        # Compute attention weights with sparse regularization
        attn_weights = torch.softmax(self.dynamic_adj_matrix, dim=-1)
        sparse_attn_weights = attn_weights * (attn_weights > 0).float()
        sparse_attn_weights = sparse_attn_weights.to(self.device)

        # Combine fixed and dynamic adjacency matrices
        adj_matrix = self.fixed_adj_matrix.to(self.device) + sparse_attn_weights
        attn_mask = (1 - adj_matrix) * -1e9

        attn_output, _ = self.attention(x, x, x, attn_mask=attn_mask)

        encoder_output = self.transformer_encoder(attn_output)
        encoder_output = self.dropout1(encoder_output)

        outputs = [output_layer(encoder_output[:, i, :]) for i, output_layer in enumerate(self.output_layers)]
        return torch.cat(outputs, dim=1)