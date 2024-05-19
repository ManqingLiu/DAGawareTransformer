import torch.nn as nn
import torch
from typing import Dict


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        # Copied and modified from PyTorch source to capture attention
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn.detach()  # Make sure to detach to prevent memory leaks


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


        self.attn_mask = ~(self.adj_matrix.bool().T)

        #print(self.attn_mask)

        #self.embedding = nn.ModuleDict({
        #    node: nn.Embedding(self.input_nodes[node]['num_categories'], self.embedding_dim)
        #    for node in self.input_nodes.keys()
        #})

        self.embedding = nn.ModuleDict({
            node.replace('.', '_'): nn.Embedding(self.input_nodes[node]['num_categories'], self.embedding_dim)
            for node in self.input_nodes.keys()
        })


        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim,
                                                        nhead=self.num_heads,
                                                        dropout=self.dropout_rate,
                                                        batch_first=True)

        '''
        self.encoder_layer = CustomTransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        '''
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.output_head = nn.ModuleDict({
            node: nn.Linear(self.embedding_dim, self.output_nodes[node]['num_categories'])
            for node in self.output_nodes.keys()
        })


    def forward(self, x, mask=None):
        #embeddings = [self.embedding[node.replace('.', '_')](x[node]) for node in self.node_ids.keys()]
        embeddings = [self.embedding[node.replace('.', '_')](x[node].long()) for node in self.node_ids.keys()]
        x = torch.stack(embeddings).squeeze(2)
        x = x.view(x.size(1), x.size(0), x.size(2))


        if mask:
            attn_mask = self.attn_mask.repeat(x.size(0) * self.num_heads, 1, 1)
            attn_mask = attn_mask.to(x.device)
            x = self.encoder(x, mask=attn_mask)
        else:
            x = self.encoder(x)
        node_outputs = {}
        for node_name in self.output_nodes.keys():
            node_id = self.node_ids[node_name]
            node_outputs[node_name] = self.output_head[node_name](x[:, node_id, :])

        return node_outputs

        '''
        if mask:
            attn_mask = self.attn_mask.repeat(x.size(0) * self.num_heads, 1, 1)
            attn_mask = attn_mask.to(x.device)

            attn_weights = []

            # Pass through each layer of the encoder
            for layer in self.encoder.layers:
                x, weights = layer(x, src_mask=attn_mask)  # Pass the correct arguments
                attn_weights.append(weights)

            # Process the output through the output heads
            node_outputs = {}
            for node_name in self.output_nodes.keys():
                node_id = self.node_ids[node_name]
                node_outputs[node_name] = self.output_head[node_name](x[:, node_id, :])

            return node_outputs, attn_weights
        else:
            x = self.encoder(x)
            node_outputs = {}
            for node_name in self.output_nodes.keys():
                node_id = self.node_ids[node_name]
                node_outputs[node_name] = self.output_head[node_name](x[:, node_id, :])

            return node_outputs
        '''


def causal_loss_fun(outputs, labels, return_items=True):

    loss = []
    batch_items = {}
    for output_name in outputs.keys():
        output = outputs[output_name]
        label = labels[output_name].squeeze()
        batch_loss = nn.functional.cross_entropy(output, label)
        if return_items:
            batch_items[output_name] = batch_loss.item()
        loss.append(batch_loss)

    loss = sum(loss) / len(loss)
    if return_items:
        return loss, batch_items
    else:
        return loss