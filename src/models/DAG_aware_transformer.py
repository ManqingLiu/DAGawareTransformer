##### DAG aware transformer - Model architecture
import torch
import torch.nn as nn
from config import *

##### DAG aware transformer - Model architecture
import torch
import torch.nn as nn

from typing import List, Union, Dict

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TabularBERT(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 embedding_dim: int,
                 nhead: int,
                 dag: List,
                 batch_size: int,
                 categorical_dims: List,
                 continuous_dims: List,
                 device: torch.device,
                 dropout_rate: float):
        super(TabularBERT, self).__init__()
        self.device = device

        # Total number of features
        # total_features = len(categorical_dims) + len(continuous_dims)

        # Embeddings for all features
        # TODO: Check if the embeddings are being created correctly
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embedding_dim) for dim in categorical_dims + continuous_dims
        ])
        #self.embeddings = nn.ModuleList([nn.Embedding(2+NUM_BINS, embedding_dim)])
        # print(self.embeddings)
        # batch_first will get me (batch, seq, feature)
        self.attention = nn.MultiheadAttention(embedding_dim, nhead, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True) ## check this line
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)  # hyperparameter: num_layers
        #self.dropout2 = nn.Dropout(dropout_rate)
        #self.decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True)
        #self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1)

        self.adj_matrix = torch.zeros(num_nodes, num_nodes)
        for edge in dag:
            self.adj_matrix[edge[0], edge[1]] = 1
        # change self.adj_mask to boolean matrix where if adj_matrix = 1, return False, else True
        self.adj_matrix = self.adj_matrix+torch.eye(num_nodes).unsqueeze(0)
        #print(self.adj_matrix)
        self.adj_mask = ~(self.adj_matrix.bool())
        #print(self.adj_mask)
        #self.adj_mask = (1 - self.adj_matrix) * -1e9
        self.adj_mask = self.adj_mask.to(device)

        self.output_layers = nn.ModuleList([nn.Linear(embedding_dim, 1) for _ in range(num_nodes)])
        self.nhead = nhead
        self.batch_size = batch_size

    def forward(self, x, use_attn_mask: bool = True):
        # Split the tensor into individual features and apply corresponding embeddings
        '''
        print(len(self.embeddings))
        print((x.size(1)))
        for i in range(x.size(1)):
            # Debug prints to check the values at each iteration
            print(f"Processing feature/column: {i}")
            print(f"Current size of self.embeddings: {len(self.embeddings)}")
            print(f"Shape of x: {x.shape}")
            print(f"Accessing x[:, {i}]. Shape of sliced x: {x[:, i].shape}")

            # Attempt to access and process the embedding
            try:
                embedding_output = self.embeddings[i](x[:, i])
                print(f"Successfully processed embedding for feature {i}")
            except Exception as e:
                print(f"Error processing embedding for feature {i}: {e}")
                break  # Break out of the loop on error to prevent further attempts that will also likely fail
        '''
        embeddings = [self.embeddings[i](x[:, i]) for i in range(x.size(1))]

        # Assuming all tensors in 'embeddings' have shape [batch_size, embedding_dim]
        x = torch.stack(embeddings)

        # 'concatenated_embeddings' now has a shape of [seq_len, batch_size, embedding_dim]
        # print(x.size()) # 20, 32, 128 (seq_len, batch_size, embedding_dim)
        # x = torch.cat(embeddings, dim=1)
        # print(x.size())
        x = x.permute(1, 0, 2)  # Transformer expects batch_size, seq_len, input_dim
        # print(x.size())
        # Compute the actual batch size
        actual_batch_size = x.size(0)
        if use_attn_mask:
            attn_mask = self.adj_mask.repeat(actual_batch_size * self.nhead, 1, 1)
        else:
            attn_mask = None
        # print(self.adj_mask.size())
        # print(attn_mask.size())
        #attn_output, _ = self.attention(x, x, x, attn_mask=attn_mask)
        #attn_output, _ = self.attention(x, x, x, attn_mask=self.adj_mask)
        #encoder_output = self.transformer_encoder(attn_output)
        encoder_output = self.transformer_encoder(x, mask=attn_mask)
        encoder_output = self.dropout1(encoder_output)
        outputs = [output_layer(encoder_output[:, i, :]) for i, output_layer in enumerate(self.output_layers)]
        return torch.cat(outputs, dim=1)

class DAGTransformer(nn.Module):
    '''
    This is a transformer module that takes in the adjacency matrix of the graph
    '''
    def __init__(self,
                 dag: Dict,
                 embedding_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout_rate: float = 0.1):

        super(DAGTransformer, self).__init__()
        self.input_nodes = dag['input_nodes']
        self.output_nodes = dag['output_nodes']
        self.edges = dag['edges']
        self.node_ids = dag['node_ids']
        self.id2node = {v: k for k, v in self.node_ids.items()}

        self.num_nodes = len(self.node_ids.keys())
        #self.num_prediction_nodes = self.num_data_nodes
        #self.num_total_nodes = self.num_data_nodes + self.num_prediction_nodes
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.adj_matrix = torch.zeros(self.num_nodes, self.num_nodes)
        for source_node_name in self.edges.keys():
            source_node_id = self.node_ids[source_node_name]
            for target_node in self.edges[source_node_name]:
                target_node_id = self.node_ids[target_node]
                self.adj_matrix[source_node_id, target_node_id] = 1

        # change self.adj_mask to boolean matrix where if adj_matrix = 1, return False, else True
        self.adj_matrix = self.adj_matrix

        self.attn_mask = ~(self.adj_matrix.bool().T)

        self.embedding = nn.ModuleDict({
            node: nn.Embedding(self.input_nodes[node]['num_categories'], self.embedding_dim)
            for node in self.input_nodes.keys()
        })

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim,
                                                        nhead=self.num_heads,
                                                        batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.output_head = nn.ModuleDict({
            node: nn.Linear(self.embedding_dim, self.output_nodes[node]['num_categories'])
            for node in self.output_nodes.keys()
        })
    def forward(self, x):
        embeddings = [self.embedding[node](x[node]) for node in self.node_ids.keys()]
        x = torch.stack(embeddings).squeeze(2)
        x = x.view(x.size(1), x.size(0), x.size(2))

        attn_mask = self.attn_mask.repeat(x.size(0) * self.num_heads, 1, 1)
        x = self.encoder(x, mask=attn_mask)
        node_outputs = {}
        for node_name in self.output_nodes.keys():
            node_id = self.node_ids[node_name]
            node_outputs[node_name] = self.output_head[node_name](x[:, node_id, :])

        return node_outputs