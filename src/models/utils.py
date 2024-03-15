
import torch
import numpy as np
import math
import pandas as pd
import random



def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def rmse(value1, value2):
    squared_difference = (value1 - value2) ** 2
    root_mean_square_error = math.sqrt(squared_difference)
    return root_mean_square_error

'''
def generate_dag_edges(all_features):
    """
    Generates edges for a DAG based on specified rules.

    Parameters:
    - all_features: List of all feature names in the dataset.

    Returns:
    - edges: List of tuples representing the edges in the DAG.
    """
    # Identify key feature indices
    t_index = all_features.index('t')
    t_hat_index = all_features.index('t_hat')
    #y_bin_index = all_features.index('y_bin')
    y_hat_bin_index = all_features.index('y_hat_bin')

    # Initialize the list of edges
    edges = []

    # Add edge from treatment to outcome_hat_binned
    edges.append((t_index, y_hat_bin_index))

    # Add edges from all original covariates to treatment_hat and outcome_hat_binned,
    # but exclude edges from outcome_binned to both treatment_hat and outcome_hat_binned
    for i, feature in enumerate(all_features):
        if "_hat" not in feature and feature != 'y_bin':  # Exclude "_hat" features and outcome_binned
            if feature != 't':  # Avoid adding edge from treatment to treatment_hat
                edges.append((i, t_hat_index))
            edges.append((i, y_hat_bin_index))

    return edges
'''


def generate_dag_edges(all_features):
    """
    Generates edges for a DAG based on specified rules.

    Parameters:
    - all_features: List of all feature names in the dataset.

    Returns:
    - edges: Set of tuples representing the edges in the DAG, ensuring no duplicates.
    """
    # Initialize the list of edges as a set to prevent duplicates
    edges = set()

    # Identify key feature indices
    t_index = all_features.index('t')
    t_hat_index = all_features.index('t_hat')
    y_hat_bin_index = all_features.index('y_hat_bin')

    # Add edge from treatment to outcome_hat_binned
    edges.add((t_index, y_hat_bin_index))

    # Add edges from all original covariates to treatment_hat and outcome_hat_binned,
    # but exclude edges from outcome_binned to both treatment_hat and outcome_hat_binned
    for i, feature in enumerate(all_features):
        if "_hat" not in feature and feature != 'y_bin' and feature != 't':  # Exclude "_hat" features, outcome_binned, and 't'
            edges.add((i, t_hat_index))
            edges.add((i, y_hat_bin_index))

    # Check if 'u_hat_bin' is in the list of all features
    if 'u_hat_bin' in all_features:
        u_hat_bin_index = all_features.index('u_hat_bin')
        # Add edges from 'u_hat_bin' to 't_hat' and 'y_hat_bin'
        edges.add((u_hat_bin_index, t_hat_index))
        edges.add((u_hat_bin_index, y_hat_bin_index))

    return list(edges)


class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, binary_features, feature_names,
                 criterion_binary, criterion_continuous, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.binary_features = binary_features
        self.feature_names = feature_names
        self.criterion_binary = criterion_binary
        self.criterion_continuous = criterion_continuous
        self.device = device


    def train(self, optimizer, loader):
        self.model.train()

        for batch in loader:  # Process each batch
            batch_gpu = batch[0].to(self.device)
            optimizer.zero_grad()
            train_losses = self._compute_losses(batch_gpu, optimizer=optimizer, train=True)
            return train_losses

    def validate(self, loader):
        torch.cuda.empty_cache()
        self.model.eval()
        feature_names = self.feature_names
        #feature_names.remove('u_hat_bin')
        total_loss_count = {feature_name + '_hat': 0 for feature_name in feature_names if '_hat' not in feature_name}
        # get length of total_loss_count (this is a dictionary)
        # print("Length of total_loss_count:", len(total_loss_count))
        with torch.no_grad():
            for batch in loader:
                batch_gpu = batch[0].to(self.device)
                losses = self._compute_losses(batch_gpu, train=False)
                for key, loss in losses.items():
                    total_loss_count[key] += loss

        # Average the losses over all batches
        num_batches = len(loader)
        avg_losses = {key: total_loss / num_batches for key, total_loss in total_loss_count.items()}
        return avg_losses

    #def test(self):
    #    self.model.eval()
    #    test_losses = self._evaluate(self.test_loader)
    #    return test_losses


    def _compute_losses(self, batch_gpu, optimizer=None, train=True):
        losses = {}
        num_feature_pairs = len(self.feature_names)
        feature_names = self.feature_names
        #print(feature_names)

        for i in range(0, num_feature_pairs - 1, 2):
            feature_name = feature_names[i]
            #if '_hat' not in feature_name:  # Process only original feature names
            feature_name_hat = feature_name + '_hat'
            output = self.model(batch_gpu)[:, i + 1].unsqueeze(1)
            true_values = batch_gpu[:, i].unsqueeze(1)

            if feature_name in self.binary_features:
                criterion = self.criterion_binary
            else:
                criterion = self.criterion_continuous

            loss = criterion(output.float(), true_values.float())
            if train:
                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()
            losses[feature_name_hat] = loss.item()
        return losses


    def get_predictions(self, loader):
        """Generates predictions for the data provided by the given DataLoader."""
        self.model.eval()
        all_predictions = []
        feature_names = self.feature_names

        with torch.no_grad():
            for batch in loader:
                # Assuming the first element of the batch is the input data
                data = batch[0].to(self.device)
                predictions = self.model(data)

                # Store predictions for this batch
                all_predictions.append(predictions.cpu())  # Move predictions to CPU

        # Concatenate all batch predictions
        all_predictions = torch.cat(all_predictions, dim=0)

        # Convert predictions into a dictionary format with feature names
        # num_feature_pairs = len(self.feature_names)
        # pred_dict = {self.feature_names[i]: all_predictions[:, i].numpy() for i in range(1, num_feature_pairs, 2)}
        num_feature_pairs = len(feature_names)
        pred_dict = {feature_names[i]: all_predictions[:, i].numpy() for i in range(1, num_feature_pairs, 2)}

        return pred_dict


'''
def get_predictions(self, loader):
        """Generates predictions for the data provided by the given DataLoader."""
        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for batch in loader:
                # Assuming the first element of the batch is the input data
                data = batch[0].to(self.device)
                predictions = self.model(data)

                # Store predictions for this batch
                all_predictions.append(predictions.cpu())  # Move predictions to CPU

        # Concatenate all batch predictions
        all_predictions = torch.cat(all_predictions, dim=0)

        # Convert predictions into a dictionary format with feature names
        num_feature_pairs = len(self.feature_names)
        pred_dict = {self.feature_names[i]: all_predictions[:, i].numpy() for i in range(1, num_feature_pairs, 2)}

        return pred_dict

'''

def IPTW_stabilized(t, y, pred_t):
    """
    Calculate the treatment effect using Stabilized Inverse Probability of Treatment Weighting (IPTW) without trimming.

    Parameters:
    - t (array-like): Treatment indicator variable (1 for treated, 0 for untreated).
    - y (array-like): Outcome variable.
    - pred_t (array-like): Predicted propensity score, e(w), for receiving treatment.

    Returns:
    - tau_hat (float): Estimated treatment effect, stabilized.
    """

    # Ensure inputs are numpy arrays for element-wise operations
    #t = np.array(t)
    #y = np.array(y)
    #pred_t = np.array(pred_t)

    # Calculate the proportion of treated and untreated
    pt_1 = np.mean(t)
    pt_0 = 1 - pt_1

    # Calculate stabilized weights without applying trimming
    weights_treated = pt_1 / pred_t
    weights_untreated = pt_0 / (1 - pred_t)

    # Calculate the numerator for each observation
    numerator = (t * y * weights_treated) - ((1 - t) * y * weights_untreated)

    # Sum over all observations and divide by the number of observations to get tau_hat
    tau_hat = np.sum(numerator) / len(t)

    return tau_hat


def AIPW(treatment, treatment_hat, outcome, outcome_hat):
    """
    Calculates the full sample AIPW estimator of E[Y^a].

    Parameters:
    - treatment: np.array, actual treatment received.
    - treatment_hat: np.array, predicted probability of receiving the treatment.
    - outcome: np.array, actual outcome observed.
    - outcome_hat: np.array, predicted outcome.

    Returns:
    - float, E[Y^a].
    """

    if len(treatment) != len(treatment_hat) or len(outcome) != len(outcome_hat) or len(treatment) != len(outcome):
        raise ValueError("All input arrays must have the same length.")

    Y_a = outcome_hat + (treatment / treatment_hat) * (outcome - outcome_hat)
    return np.mean(Y_a)





