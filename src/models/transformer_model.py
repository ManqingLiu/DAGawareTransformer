import torch.nn as nn
import torch
from typing import Dict
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import wandb

from src.dataset import CausalDataset
from src.utils import predict_function, replace_column_values
from src.train.lalonde.train_metrics import calculate_val_metrics
from src.train.ihdp.train_metrics import calculate_val_metrics_ihdp
from src.evaluate.lalonde.evaluate_metrics import calculate_test_metrics
from src.evaluate.ihdp.evaluate_metrics import calculate_test_metrics_ihdp
from src.train.acic.train_metrics import calculate_val_metrics_acic
from src.evaluate.acic.evaluate_metrics import calculate_test_metrics_acic


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
                                                        dim_feedforward=self.embedding_dim,  # TODO: this used to be the default (2048) so the flow was 20 -> 2048 -> 20 dim
                                                        activation='relu',
                                                        batch_first=True)

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.output_head = nn.ModuleDict({
            node: nn.Sequential(
            nn.Linear(self.embedding_dim, self.output_nodes[node]['num_categories']),
            nn.Softmax(dim=1)
            )
            for node in self.output_nodes.keys()
        })

    def forward(self, x, mask=None):
        # TODO: embeddings are dim 20
        embeddings = [self.embedding[node.replace('.', '_')](x[node].long()) for node in self.node_ids.keys()]
        x = torch.stack(embeddings).squeeze(2)
        x = x.permute(1, 0, 2)

        if mask=="True":
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
            imbalance_loss_weight: float,
            random_seed: int = None
    ) -> nn.Module:

        train_config = config[estimator]["training"]

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

        rmse_cfcv = float('inf')
        rmse_ipw = float('inf')


        for epoch in range(train_config["num_epochs"]):
            print(f"Epoch: {epoch}")
            model.train()
            for batch_ix, (batch_raw, batch_binned) in enumerate(train_dataloader):
                opt.zero_grad()
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

                t = batch_raw['t'].to(device)
                y = batch_raw['y'].to(device)
                e = transformed_outputs['t'].to(device)
                y_ = torch.squeeze(transformed_outputs['y']).to(device)
                h_rep_norm = h_rep / safe_sqrt(torch.sum(h_rep ** 2, dim=1, keepdim=True))
                h_rep_norm = h_rep_norm.to(device)


                # Use CFR loss function
                batch_loss, batch_items = CFR_loss(t, e, y, y_, h_rep_norm,
                                                   imbalance_loss_weight, return_items=True,
                                                   eps=train_config["eps"])

                batch_loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                # print epoch number and batch loss
                #print(f"Epoch: {epoch}, Batch: {batch_ix}, Loss: {batch_loss.item()}")
                wandb.log({f"Train: counterfactual loss": batch_loss.item()})

            model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch_raw_val, batch_binned_val in val_dataloader:
                    batch_val = {k: v.to(device) for k, v in batch_binned_val.items()}
                    outputs_val = model(batch_val, mask=train_config["dag_attention_mask"])

                    transformed_outputs_val = {}
                    for output_name, output in outputs_val.items():
                        softmax_output = torch.softmax(output, dim=1)
                        if output_name == 'y':
                            pred_y = torch.sum(softmax_output * bin_left_edges[output_name].to(device), dim=1,
                                               keepdim=True)
                            transformed_outputs_val[output_name] = pred_y
                        else:
                            transformed_outputs_val[output_name] = softmax_output[:, 1]
                            h_rep = output

                    t = batch_raw_val['t'].to(device)
                    y = batch_raw_val['y'].to(device)
                    e = transformed_outputs_val['t'].to(device)
                    y_ = torch.squeeze(transformed_outputs_val['y']).to(device)
                    h_rep_norm = h_rep / safe_sqrt(torch.sum(h_rep ** 2, dim=1, keepdim=True))
                    h_rep_norm = h_rep_norm.to(device)

                    # Use CFR loss function
                    val_batch_loss, val_batch_items = CFR_loss(t, e, y, y_, h_rep_norm, imbalance_loss_weight,
                                                               return_items=True, eps=train_config["eps"])

                    val_loss += val_batch_loss.item()
                    val_loss_avg = val_loss / len(val_dataloader)
            wandb.log({f"Val: counterfactual loss": val_loss_avg})

            predictions_val, metrics_val, metrics_test = model.predict(model,
                                            data_name,
                                            val_data,
                                            pseudo_ate_data,
                                            sample_id,
                                            dag=dag,
                                            train_config=train_config,
                                            random_seed=random_seed,
                                            prefix="Test",
                                            estimator=estimator)

            for metric_name, metric_value in metrics_test.items():
                print(f"Epoch: {epoch}: {metric_name}: {metric_value}")

            rmse_cfcv = metrics_test.get("Test: NRMSE for AIPW", float('inf'))
            rmse_ipw = metrics_test.get("Test: NRMSE for IPW", float('inf'))
            
            if data_name == "lalonde_cps":
                wandb.log(
                calculate_val_metrics(
                    predictions_val,
                    pseudo_ate_data,
                    sample_id,
                    prefix="Val",
                    prop_score_threshold=train_config["prop_score_threshold"]
                ))
                wandb.log(
                    metrics_test
                )

            if data_name == "lalonde_psid":
                wandb.log(
                    calculate_val_metrics(
                        predictions_val,
                        pseudo_ate_data,
                        sample_id,
                        prefix="Val",
                        prop_score_threshold=train_config["prop_score_threshold"]
                    ))
                wandb.log(
                    metrics_test
                )

            elif data_name == "ihdp":
                wandb.log(
                calculate_val_metrics_ihdp(
                    predictions_val,
                    pseudo_ate_data,
                    prefix="Val",
                    prop_score_threshold=train_config["prop_score_threshold"]
                ))
            elif data_name == "acic":
                wandb.log(
                calculate_val_metrics_acic(
                    predictions_val,
                    pseudo_ate_data,
                    prefix="Val",
                    prop_score_threshold=train_config["prop_score_threshold"]
                ))
                wandb.log(
                    metrics_test
                )


        print(f"RMSE for CFCV: {rmse_cfcv}")
        print(f"RMSE for IPW: {rmse_ipw}")
        return model, rmse_cfcv, rmse_ipw

    @staticmethod
    def predict(model,
                data_name,
                data,
                pseudo_ate_data,
                sample_id,
                dag,
                train_config: Dict,
                random_seed: int,
                prefix: str = "Test",
                estimator: str= "aipw"):
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

        predictions_t = predict_function(model, train_config, dataloader)['t']
        # convert predictions_t to dataframe with column name t_prob
        predictions_t = pd.DataFrame(predictions_t, columns=['t_prob'])
        predictions_y = predict_function(model, train_config, dataloader)['y']
        # convert predictions_y to dataframe with column name pred_y
        predictions_y = pd.DataFrame(predictions_y, columns=['pred_y'])
        predictions_y0 = predict_function(model, train_config, dataloader_A0)['y']
        # convert predictions_y0 to dataframe with column name pred_y_A0
        predictions_y0 = pd.DataFrame(predictions_y0, columns=['pred_y_A0'])
        predictions_y1 = predict_function(model, train_config, dataloader_A1)['y']
        # convert predictions_y1 to dataframe with column name pred_y_A1
        predictions_y1 = pd.DataFrame(predictions_y1, columns=['pred_y_A1'])

        final_predictions = pd.concat(
            [data,  predictions_t["t_prob"], predictions_y["pred_y"], predictions_y0["pred_y_A0"], predictions_y1["pred_y_A1"]],
            axis=1,
        )

        if data_name == "lalonde_cps":
            metrics_val = calculate_val_metrics(
                                            final_predictions,
                                            pseudo_ate_data,
                                            sample_id,
                                            prefix="Val",
                                            prop_score_threshold=train_config["prop_score_threshold"])

            metrics_test = calculate_test_metrics(final_predictions,
                                                  prop_score_threshold=train_config["prop_score_threshold"],
                                                  prefix=prefix,
                                                  estimator=estimator)

        elif data_name == "lalonde_psid":
            metrics_val = calculate_val_metrics(
                                            final_predictions,
                                            pseudo_ate_data,
                                            sample_id,
                                            prefix="Val",
                                            prop_score_threshold=train_config["prop_score_threshold"])

            metrics_test = calculate_test_metrics(final_predictions,
                                                  prop_score_threshold=train_config["prop_score_threshold"],
                                                  prefix=prefix,
                                                  estimator=estimator)
        elif data_name == "ihdp":
            metrics_val = calculate_val_metrics_ihdp(
                                            final_predictions,
                                            pseudo_ate_data,
                                            prefix="Val",
                                            prop_score_threshold=train_config["prop_score_threshold"])

            metrics_test = calculate_test_metrics_ihdp(final_predictions,
                                                       data['ite'],
                                                  prop_score_threshold=train_config["prop_score_threshold"],
                                                  prefix=prefix)

        elif data_name == "acic":
            metrics_val = calculate_val_metrics_acic(
                                            final_predictions,
                                            pseudo_ate_data,
                                            prefix="Val",
                                            prop_score_threshold=train_config["prop_score_threshold"])

            metrics_test = calculate_test_metrics_acic(final_predictions,
                                                       data['mu1']-data['mu0'],
                                                  prop_score_threshold=train_config["prop_score_threshold"],
                                                  prefix=prefix)




        return final_predictions, metrics_val, metrics_test



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


def safe_sqrt(x, lbound=1e-5):
    ''' Numerically safe version of PyTorch sqrt '''
    return torch.sqrt(torch.clamp(x, min=lbound))

def ensure_non_empty(X, eps=1e-6):
    """ Ensures that a tensor is non-empty by adding a small epsilon tensor if necessary """
    if X.size(0) == 0:
        X = torch.full((1, X.size(1)), eps, device=X.device)
    return X

def wasserstein(X, t, p, lam=1, its=50, backpropT=False, eps=1e-6):
    """ Returns the Wasserstein distance between treatment groups """
    Xt = X[t == 1]
    Xc = X[t == 0]
    nc = float(Xc.size(0))
    nt = float(Xt.size(0))

    # Add eps to Xt or Xc if they are empty
    Xt = ensure_non_empty(Xt, eps)
    Xc = ensure_non_empty(Xc, eps)

    # Compute distance matrix
    M = safe_sqrt(pdist2sq(Xt, Xc), eps)

    # Estimate lambda and delta
    M_mean = torch.mean(M)
    # Calculate the dropout probability and cap it between 0 and 1
    dropout_prob = np.clip(10 / (nc * nt + eps), 0.0, 1.0)
    # Apply dropout with the capped probability
    M_drop = torch.nn.functional.dropout(M, p=dropout_prob)

    delta = M.max().detach()
    eff_lam = (lam / M_mean).detach()

    # Compute new distance matrix
    Mt = M
    row = delta * torch.ones((1, M.size(1))).to(X.device)
    col = torch.cat([delta * torch.ones((M.size(0), 1)).to(X.device), torch.zeros((1, 1)).to(X.device)], dim=0)
    Mt = torch.cat([M, row], dim=0)
    Mt = torch.cat([Mt, col], dim=1)

    # Compute marginal vectors
    a = torch.cat([p * torch.ones((Xt.size(0), 1)).to(X.device) / (nt+0.01), (1 - p) * torch.ones((1, 1)).to(X.device)], dim=0)
    b = torch.cat([(1 - p) * torch.ones((Xc.size(0), 1)).to(X.device) / nc, p * torch.ones((1, 1)).to(X.device)], dim=0)

    # Compute kernel matrix
    Mlam = eff_lam * Mt
    K = torch.exp(-Mlam) + eps
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
             eps: float=1e-6,
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
        eps: float
            Small value to ensure numerical stability.
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = t.to(device)
    e = e.to(device)
    y = y.to(device)
    y_ = y_.to(device)
    w_t = t * (1 - e) / e
    w_c = (1 - t) * e / (1 - e)
    sample_weight = torch.clamp(w_t + w_c, min=0.0, max=100.0)
    ''' Construct factual loss function '''
    risk = torch.mean(sample_weight * torch.sqrt(torch.abs(y-y_)))

    p_ipm = 0.5
    p_ipm = torch.tensor(p_ipm).to(device)
    ''' Compute imbalance error '''
    imb_dist, imb_mat = wasserstein(h_rep_norm, t, p_ipm,
                                    its=wass_iterations, lam=wass_lambda, backpropT=wass_bpt, eps=eps)
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

