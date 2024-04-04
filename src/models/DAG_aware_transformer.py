##### DAG aware transformer - Model architecture
import torch
import torch.nn as nn

##### DAG aware transformer - Model architecture
import torch
import torch.nn as nn

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TabularBERT(nn.Module):
    def __init__(self, num_nodes, embedding_dim, nhead, dag, batch_size, categorical_dims, continuous_dims, device, dropout_rate):
        super(TabularBERT, self).__init__()
        self.device = device

        # Total number of features
        # total_features = len(categorical_dims) + len(continuous_dims)

        # Embeddings for all features
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embedding_dim) for dim in categorical_dims + continuous_dims
        ])
        # print(self.embeddings)
        # batch_first will get me (batch, seq, feature)
        self.attention = nn.MultiheadAttention(embedding_dim, nhead, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True) ## check this line
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)  # hyperparameter: num_layers
        #self.dropout2 = nn.Dropout(dropout_rate)
        #self.decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True)
        #self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1)

        self.adj_matrix = torch.zeros(num_nodes, num_nodes)
        for edge in dag:
            self.adj_matrix[edge[0], edge[1]] = 1
        self.adj_mask = (1 - self.adj_matrix) * -1e9
        self.adj_mask = self.adj_mask.to(device)

        self.output_layers = nn.ModuleList([nn.Linear(embedding_dim, 1) for _ in range(num_nodes)])
        self.nhead = nhead
        self.batch_size = batch_size

    def forward(self, x):
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
        # batch_size = x.size(1)
        # attn_mask = self.adj_mask.repeat(self.batch_size * self.nhead, 1, 1)
        # print(self.adj_mask.size())
        # print(attn_mask.size())
        #attn_output, _ = self.attention(x, x, x, attn_mask=attn_mask)
        #attn_output, _ = self.attention(x, x, x, attn_mask=self.adj_mask)
        #encoder_output = self.transformer_encoder(attn_output)
        encoder_output = self.transformer_encoder(x, mask=self.adj_mask)
        # encoder_output = self.dropout1(encoder_output)


        outputs = [output_layer(encoder_output[:, i, :]) for i, output_layer in enumerate(self.output_layers)]
        return torch.cat(outputs, dim=1)