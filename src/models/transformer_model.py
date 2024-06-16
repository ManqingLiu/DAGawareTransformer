import torch.nn as nn
import torch
from typing import Dict
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import wandb

from src.dataset import CausalDataset
from src.utils import predict_function, replace_column_values


class DAGTransformer(nn.Module):
    '''
    This is a transformer module that takes in the adjacency matrix of the graph
    '''
    def __init__(self,
                 dag: Dict,
                 embedding_dim: int,
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

        self.embedding = nn.ModuleDict({
            node.replace('.', '_'): nn.Embedding(self.input_nodes[node]['num_categories'], self.embedding_dim)
            for node in self.input_nodes.keys()
        })


        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim,
                                                        nhead=self.num_heads,
                                                        dropout=self.dropout_rate,
                                                        activation='relu',
                                                        batch_first=True)

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.output_head = nn.ModuleDict({
            node: nn.Linear(self.embedding_dim, self.output_nodes[node]['num_categories'])
            for node in self.output_nodes.keys()
        })


    def forward(self, x, mask=None):
        if (x['t'] == 0).all():
            embeddings = [self.embedding[node.replace('.', '_')](x[node].long()) for node in self.node_ids.keys()]
            x_emb = torch.stack(embeddings).squeeze(2)
            x_emb = x_emb.permute(1, 0, 2)
            if mask:
                attn_mask = self.attn_mask.repeat(x_emb.size(0) * self.num_heads, 1, 1)
                attn_mask = attn_mask.to(x_emb.device)
                x = self.encoder(x_emb, mask=attn_mask)
            else:
                x = self.encoder(x_emb)
        else:
            # Create a mask for instances where 't' is 0
            t0 = x['t'] == 0
            # Use the mask to select the appropriate instances from 'x'
            x_control = {k: v[t0] for k, v in x.items()}
            # Modify the 'embeddings_control' line to use 'x_masked' instead of 'x'
            embeddings_control = [self.embedding[node.replace('.', '_')](x_control[node].long()) for node in
                                  self.node_ids.keys()]
            t1 = x['t'] == 1
            x_treatment = {k: v[t1] for k, v in x.items()}
            embeddings_treatment = [self.embedding[node.replace('.', '_')](x_treatment[node].long()) for node in
                                    self.node_ids.keys()]

            x_control_emb = torch.stack(embeddings_control).squeeze(2)
            x_treatment_emb = torch.stack(embeddings_treatment).squeeze(2)

            x_control_emb = x_control_emb.permute(1, 0, 2)
            x_treatment_emb = x_treatment_emb.permute(1, 0, 2)

            if mask:
                attn_mask_control = self.attn_mask.repeat(x_control_emb.size(0) * self.num_heads, 1, 1)
                attn_mask_control = attn_mask_control.to(x_control_emb.device)
                x_control = self.encoder(x_control_emb, mask=attn_mask_control)
                attn_mask_treatment = self.attn_mask.repeat(x_treatment_emb.size(0) * self.num_heads, 1, 1)
                attn_mask_treatment = attn_mask_treatment.to(x_treatment_emb.device)
                x_treatment = self.encoder(x_treatment_emb, mask=attn_mask_treatment)
            else:
                x_control = self.encoder(x_control_emb)
                x_treatment = self.encoder(x_treatment_emb)

            x = torch.cat([x_control, x_treatment], dim=0)


        node_outputs = {}
        for node_name in self.output_nodes.keys():
            node_id = self.node_ids[node_name]
            node_outputs[node_name] = self.output_head[node_name](x[:, node_id, :])

        return node_outputs

    def _train(
            self,
            model: nn.Module,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            train_config: Dict,
            imbalance_loss_weight: float
    ) -> nn.Module:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = model.to(device)

        opt = torch.optim.AdamW(
            model.parameters(),
            weight_decay=train_config["weight_decay"],
            lr=train_config["learning_rate"],
        )

        bin_left_edges = {k: torch.tensor(v[:-1], dtype=torch.float32).to(device) for k, v in
                          train_dataloader.dataset.bin_edges.items()}

        for epoch in range(train_config["num_epochs"]):
            for batch_ix, (batch_raw, batch_binned) in enumerate(train_dataloader):
                opt.zero_grad()
                batch = {k: v.to(device) for k, v in batch_binned.items()}
                outputs = model(batch, mask=train_config["dag_attention_mask"])

                transformed_outputs = {}
                for output_name, output in outputs.items():
                    softmax_output = torch.softmax(output, dim=1)
                    if output_name == 'y':
                        pred_y = torch.sum(softmax_output * bin_left_edges[output_name].to(device), dim=1, keepdim=True)
                        transformed_outputs[output_name] = pred_y
                    else:
                        transformed_outputs[output_name] = softmax_output[:, 1]
                        h_rep = output

                t = batch_raw['t']
                y = batch_raw['y']
                e = transformed_outputs['t']
                y_ = torch.squeeze(transformed_outputs['y'])
                h_rep_norm = h_rep / safe_sqrt(torch.sum(h_rep ** 2, dim=1, keepdim=True))

                if epoch == 4 and batch_ix == 1:
                    print(f"batch_ix: {batch_ix}")

                # Use CFR loss function
                batch_loss, batch_items = CFR_loss(t, e, y, y_, h_rep_norm, imbalance_loss_weight, return_items=True)

                # Debugging print statements
                print(f"batch_ix: {batch_ix}, batch_loss type: {type(batch_loss)}, value: {batch_loss}")

                if torch.isnan(batch_loss):
                    raise ValueError("NaN encountered in batch_loss")

                batch_loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                #wandb.log({f"Train: counterfactual loss": batch_loss.item()})
                # Ensure batch_loss is a tensor
                if isinstance(batch_loss, torch.Tensor):
                    wandb.log({f"Train: counterfactual loss": batch_loss.item()})
                else:
                    raise TypeError(f"Expected batch_loss to be a tensor, but got {type(batch_loss)} instead.")

            model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch_raw, batch_binned in val_dataloader:
                    batch = {k: v.to(device) for k, v in batch_binned.items()}
                    outputs = model(batch, mask=train_config["dag_attention_mask"])

                    transformed_outputs = {}
                    for output_name, output in outputs.items():
                        softmax_output = torch.softmax(output, dim=1)
                        if output_name == 'y':
                            pred_y = torch.sum(softmax_output * bin_left_edges[output_name].to(device), dim=1,
                                               keepdim=True)
                            transformed_outputs[output_name] = pred_y
                        else:
                            transformed_outputs[output_name] = softmax_output[:, 1]
                            h_rep = output

                    t = batch_raw['t']
                    y = batch_raw['y']
                    e = transformed_outputs['t']
                    y_ = torch.squeeze(transformed_outputs['y'])
                    h_rep_norm = h_rep / safe_sqrt(torch.sum(h_rep ** 2, dim=1, keepdim=True))

                    # Use CFR loss function
                    val_batch_loss, val_batch_items = CFR_loss(t, e, y, y_, h_rep_norm, imbalance_loss_weight,
                                                               return_items=True)

                    val_loss += val_batch_loss.item()
                wandb.log({f"Val: counterfactual loss": val_loss / len(batch)})

            model.train()

        return model

    @staticmethod
    def predict(model,
                data,
                dag,
                train_config: Dict,
                mask: bool,
                random_seed: int):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        batch_size = train_config["batch_size"]
        model = model.to(device)

        data = data[dag['nodes']]
        dataset = CausalDataset(data, dag, random_seed)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
        )

        data_A0 = replace_column_values(data, "t", 0)
        dataset_A0 = CausalDataset(data_A0, dag, random_seed)
        dataloader_A0 = DataLoader(
            dataset_A0,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
        )

        data_A1 = replace_column_values(data, "t", 1)
        dataset_A1 = CausalDataset(data_A1, dag, random_seed)
        dataloader_A1 = DataLoader(
            dataset_A1,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
        )

        model.eval()

        predictions_t = predict_function(model, dataset, dataloader, mask)
        predictions_y0 = predict_function(model, dataset_A0, dataloader_A0, mask)
        predictions_y0 = predictions_y0.rename(columns={"pred_y": "pred_y_A0"})
        predictions_y1 = predict_function(model, dataset_A1, dataloader_A1, mask)
        predictions_y1 = predictions_y1.rename(columns={"pred_y": "pred_y_A1"})

        final_predictions = pd.concat(
            [data,  predictions_t["t_prob"], predictions_y0["pred_y_A0"], predictions_y1["pred_y_A1"]],
            axis=1,
        )

        return final_predictions



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


def pdist2sq(X, Y):
    """ Computes the squared Euclidean distance between two matrices """
    C = -2 * torch.mm(X, Y.t())
    nx = torch.sum(X ** 2, dim=1, keepdim=True)
    ny = torch.sum(Y ** 2, dim=1, keepdim=True)
    D = (C + ny.t()) + nx
    return D


def safe_sqrt(X, eps=1e-12):
    """ Computes the element-wise square root of a matrix with numerical stability """
    return torch.sqrt(X + eps)

def ensure_non_empty(X, eps=1e-12):
    """ Ensures that a tensor is non-empty by adding a small epsilon tensor if necessary """
    if X.size(0) == 0:
        X = torch.full((1, X.size(1)), eps, device=X.device)
    return X

def wasserstein(X, t, p, lam=1, its=50, backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """
    Xt = X[t == 1]
    Xc = X[t == 0]
    nc = float(Xc.size(0))
    nt = float(Xt.size(0))
    eps = 1e-12  # Small value to avoid division by zero

    # Add eps to Xt or Xc if they are empty
    Xt = ensure_non_empty(Xt, eps)
    Xc = ensure_non_empty(Xc, eps)

    # Compute distance matrix
    M = safe_sqrt(pdist2sq(Xt, Xc))

    # Estimate lambda and delta
    M_mean = torch.mean(M)
    # Calculate the dropout probability and cap it between 0 and 1
    dropout_prob = np.clip(10 / (nc * nt + 0.01), 0.0, 1.0)
    # Apply dropout with the capped probability
    M_drop = torch.nn.functional.dropout(M, p=dropout_prob)

    delta = M.max().detach()
    eff_lam = (lam / M_mean).detach()

    # Compute new distance matrix
    Mt = M
    row = delta * torch.ones((1, M.size(1)))
    col = torch.cat([delta * torch.ones((M.size(0), 1)), torch.zeros((1, 1))], dim=0)
    Mt = torch.cat([M, row], dim=0)
    Mt = torch.cat([Mt, col], dim=1)

    # Compute marginal vectors
    a = torch.cat([p * torch.ones((Xt.size(0), 1)) / (nt+0.01), (1 - p) * torch.ones((1, 1))], dim=0)
    b = torch.cat([(1 - p) * torch.ones((Xc.size(0), 1)) / nc, p * torch.ones((1, 1))], dim=0)

    # Compute kernel matrix
    Mlam = eff_lam * Mt
    K = torch.exp(-Mlam) + 1e-6  # added constant to avoid nan
    U = K * Mt
    ainvK = K / a

    u = a
    for i in range(its):
        u = 1.0 / (ainvK @ (b / (u.t() @ K).t()))

    v = b / (u.t() @ K).t()
    T = u * (v.t() * K)

    if not backpropT:
        T = T.detach()

    E = T * Mt
    D = 2 * torch.sum(E)

    return D, Mlam


def CFR_loss(t, e, y, y_, h_rep_norm,
             alpha: float,
             wass_iterations: int=10,
             wass_lambda: float=10.0,
             wass_bpt: bool=True,
             return_items: bool=True):
    """
        Compute the Counterfactual Regression (CFR) loss.

        Parameters:
        ----------
        t : torch.Tensor
            Treatment indicators, where 1 indicates treatment and 0 indicates control.
        e : torch.Tensor
            Propensity scores, i.e., the probability of receiving treatment.
        y : torch.Tensor
            Observed outcomes.
        y_ : torch.Tensor
            Predicted outcomes.
        h_rep_norm : torch.Tensor
            Normalized representation of the input features (which is normalized e).
        wass_iterations : int
            Number of iterations for the Wasserstein distance calculation.
        wass_lambda : float
            Regularization parameter for the Wasserstein distance.
        wass_bpt : bool
            Flag indicating whether to backpropagate through the transport matrix in the Wasserstein distance calculation.
        alpha : float
            Weighting factor for the imbalance error term.

        Returns:
        -------
        tot_error : torch.Tensor
            Total loss combining the factual loss and the imbalance error.

        Description:
        ------------
        This function computes the CFR loss, which is a combination of the factual loss (risk) and the imbalance error
        (imb_error). The factual loss measures the discrepancy between observed outcomes and predicted outcomes,
        weighted by sample weights. The imbalance error measures the distributional distance between treated and
        control groups in the representation space, using the Wasserstein distance.

        The function performs the following steps:
        1. Computes sample reweighting based on treatment indicators and propensity scores.
        2. Constructs the factual loss function using the reweighted squared errors between observed and predicted outcomes.
        3. Computes the imbalance error using the Wasserstein distance between treated and control groups in the
           representation space.
        4. Combines the factual loss and the imbalance error to obtain the total error.

        Example:
        --------
        tot_error = CFR_loss(t, e, y, y_, h_rep_norm, wass_iterations=50, wass_lambda=1, wass_bpt=False, alpha=0.1)
        """
    ''' Compute sample reweighting '''
    w_t = t * (1 - e) / e
    w_c = (1 - t) * e / (1 - e)
    sample_weight = torch.clamp(w_t + w_c, min=0.0, max=100.0)
    ''' Construct factual loss function '''
    risk = torch.mean(sample_weight * torch.sqrt(torch.abs(y-y_)))

    p_ipm = 0.5
    imb_dist, imb_mat = wasserstein(h_rep_norm, t, p_ipm,
                                    its=wass_iterations, lam=wass_lambda, backpropT=wass_bpt)
    imb_error = alpha * imb_dist

    ''' Total error '''
    loss_value = risk + imb_error

    if return_items:
        batch_items = {
            'loss': loss_value.item()
        }
        return loss_value, batch_items
    else:
        return loss_value

