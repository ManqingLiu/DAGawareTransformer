
import torch
import numpy as np
import math
import pandas as pd
import random
from config import *
from scipy.special import expit as sigmoid




def set_seed(seed_value=SEED_VALUE):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def rmse(value1, value2):
    squared_difference = (value1 - value2) ** 2
    root_mean_square_error = math.sqrt(squared_difference)
    return root_mean_square_error

## define a function calculating mean absolute error
def mae(value1, value2):
    absolute_difference = np.abs(value1 - value2)
    mean_absolute_error = np.mean(absolute_difference)
    return mean_absolute_error

# TODO: Might want to create more general creation and representation machinary
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
    #y_hat_bin_index = all_features.index('y_hat_bin')
    y_hat_index = next(i for i, feature in enumerate(all_features) if feature.startswith('y_hat'))
    #y_hat_index = all_features.index('y_hat')

    # Add edge from treatment to outcome_hat_binned
    #edges.add((t_index, y_hat_bin_index))
    edges.add((t_index, y_hat_index))

    # Add edges from all original covariates to treatment_hat and outcome_hat_binned,
    # but exclude edges from outcome_binned to both treatment_hat and outcome_hat_binned
    for i, feature in enumerate(all_features):
        #if "_hat" not in feature and feature != 'y_bin' and feature != 't':  # Exclude "_hat" features, outcome_binned, and 't'
        if "_hat" not in feature and 'y' not in feature and feature != 't':
            edges.add((i, t_hat_index))
            #edges.add((i, y_hat_bin_index))
            edges.add((i, y_hat_index))

    return list(edges)

    '''
    # Check if 'u_hat_bin' is in the list of all features
    if 'u_hat_bin' in all_features:
        u_hat_bin_index = all_features.index('u_hat_bin')
        # Add edges from 'u_hat_bin' to 't_hat' and 'y_hat_bin'
        edges.add((u_hat_bin_index, t_hat_index))
        edges.add((u_hat_bin_index, y_hat_bin_index))
        #edges.add((u_hat_bin_index, y_hat_index))
    '''




def save_model(model, dataset_type, loader_name, n_sample, n_epochs,
               learning_rate, weight_decay, dropout_rate, batch_size, embedding_dim, nhead):
    # write description of the model
    model_path = (f"experiments/model/{dataset_type}/model_{dataset_type}_{loader_name}_sample{n_sample}_"
                  f"epoch{n_epochs}_lr{learning_rate:.4f}_wd{weight_decay:.4f}_dr{dropout_rate:.4f}_bs{batch_size}_ed{embedding_dim}_nh{nhead}.pth")
    # Example path: "experiments/model/model_cps_train_fold1_sample100_epoch50.pth"

    # Assuming you're using PyTorch to save the model. Adjust accordingly for other frameworks.
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def save_model_stratified_std(model, treatment, dataset_type, loader_name, n_sample, n_epochs,
               learning_rate, weight_decay, dropout_rate, batch_size, embedding_dim, nhead):
    # write description of the model
    model_path = (f"experiments/model/{dataset_type}/model_treatment{treatment}_{dataset_type}_{loader_name}_sample{n_sample}_"
                  f"epoch{n_epochs}_lr{learning_rate:.4f}_wd{weight_decay:.4f}_dr{dropout_rate:.4f}_bs{batch_size}_ed{embedding_dim}_nh{nhead}.pth")
    # Example path: "experiments/model/model_cps_train_fold1_sample100_epoch50.pth"

    # Assuming you're using PyTorch to save the model. Adjust accordingly for other frameworks.
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

'''
def predict_and_create_df(model, model_path, device, trainer, loader, data, processor, feature_names):
    # Load the model
    model.load_state_dict(torch.load(model_path))

    # Assuming the trainer's model needs to be manually updated
    trainer.model = model.to(device)

    # Perform predictions
    predictions = trainer.get_predictions(loader)
    #print(predictions['y_hat_bin'])

    # Create DataFrame from predictions
    data = data.cpu()
    df = pd.DataFrame(data.numpy(), columns=feature_names)
    df['pred_y'] = processor.bin_to_original(predictions['y_hat_bin'], 'y_hat')

    #df['pred_y'] = sigmoid(predictions['y_hat'])
    df['pred_t'] = sigmoid(predictions['t_hat'])

    return df
'''

def predict_and_create_df(model, model_path, device, trainer, loader, data, processor, feature_names, dataset_type):
    # Load the model
    model.load_state_dict(torch.load(model_path))

    # Assuming the trainer's model needs to be manually updated
    trainer.model = model.to(device)

    # Perform predictions
    predictions = trainer.get_predictions(loader)

    # Create DataFrame from predictions
    data = data.cpu()
    df = pd.DataFrame(data.numpy(), columns=feature_names)

    # Check the dataset_type and perform the appropriate operation
    if dataset_type == 'twins':
        df['pred_y'] = sigmoid(predictions['y_hat'])
    elif dataset_type.startswith('lalonde'):
        df['pred_y'] = processor.bin_to_original(predictions['y_hat_bin'], 'y_hat')

    df['pred_t'] = sigmoid(predictions['t_hat'])

    return df

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

    '''
    def train(self, optimizer, loader):
        self.model.train()

        for idx, batch in loader:  # Process each batch
            #batch_gpu = batch[0].to(self.device)
            batch_gpu = batch.to(self.device)
            optimizer.zero_grad()
            train_losses = self._compute_losses(batch_gpu, optimizer=optimizer, train=True)
            return train_losses
    '''
    def train(self, optimizer, loader):
        self.model.train()
        #train_losses = {}
        optimizer.zero_grad()
        feature_names = self.feature_names
        # feature_names.remove('u_hat_bin')
        total_loss_count = {feature_name + '_hat': 0 for feature_name in feature_names if '_hat' not in feature_name}

        for idx, batch in loader:  # Process each batch
            batch_gpu = batch.to(self.device)
            losses = self._compute_losses(batch_gpu, optimizer=optimizer, train=True)
            for key, loss in losses.items():
                # print(loss)
                total_loss_count[key] += loss

        # Average the losses over all batches
        num_batches = len(loader)
        avg_losses = {key: total_loss / num_batches for key, total_loss in total_loss_count.items()}
        return avg_losses


    def validate(self, loader):
        torch.cuda.empty_cache()
        self.model.eval()
        feature_names = self.feature_names
        #feature_names.remove('u_hat_bin')
        total_loss_count = {feature_name + '_hat': 0 for feature_name in feature_names if '_hat' not in feature_name}
        # get length of total_loss_count (this is a dictionary)
        # print("Length of total_loss_count:", len(total_loss_count))
        with torch.no_grad():
            for idx, batch in loader:
                #batch_gpu = batch[0].to(self.device)
                batch_gpu = batch.to(self.device)
                losses = self._compute_losses(batch_gpu, train=False)
                for key, loss in losses.items():
                    #print(loss)
                    total_loss_count[key] += loss

        # Average the losses over all batches
        num_batches = len(loader)
        avg_losses = {key: total_loss / num_batches for key, total_loss in total_loss_count.items()}
        return avg_losses


    def _compute_losses(self, batch_gpu, optimizer=None, train=False):
        losses = {}
        num_feature_pairs = len(self.feature_names)
        feature_names = self.feature_names
        #batch_gpu = batch_gpu.to(self.device)
        model = self.model.to(self.device)

        for i in range(0, num_feature_pairs - 1, 2):
            feature_name = feature_names[i]
            #if '_hat' not in feature_name:  # Process only original feature names
            feature_name_hat = feature_name + '_hat'
            output = model(batch_gpu)[:, i + 1].unsqueeze(1)
            true_values = batch_gpu[:, i].unsqueeze(1)

            if feature_name in self.binary_features:
                criterion = self.criterion_binary
            else:
                print("feature_name:", feature_name)
                criterion = self.criterion_continuous

            loss = criterion(output.float(), true_values.float())

            if train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            losses[feature_name_hat] = loss.item()
        return losses


    def get_predictions(self, loader):
        """Generates predictions for the data provided by the given DataLoader, ordered by the original dataset order."""
        self.model.eval()
        all_indices = []  # List to store all indices
        all_predictions = []
        feature_names = self.feature_names

        with torch.no_grad():
            for idx, data in loader:
                # Extract the batch data and send to the device
                data = data.to(self.device)
                predictions = self.model(data)

                # Store indices and predictions for this batch
                all_indices.append(idx.cpu())  # Move indices to CPU
                #all_predictions.append(predictions.cpu())  # Move predictions to CPU
                all_predictions.append(predictions)


        # Concatenate all indices and batch predictions
        all_indices = torch.cat(all_indices, dim=0).to(self.device)
        all_predictions = torch.cat(all_predictions, dim=0)

        # Sort predictions based on indices to return to original order
        sorted_indices = all_indices.argsort()
        sorted_predictions = all_predictions[sorted_indices]

        num_feature_pairs = len(feature_names)
        # Assuming feature_names[0] is not an ID column and starts with actual features
        pred_dict = {feature_names[i]: sorted_predictions[:, i].cpu().numpy() for i in range(num_feature_pairs)}

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

def IPTW_unstabilized(t, y, pred_t):
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

    # Calculate stabilized weights without applying trimming
    weights_treated = 1 / pred_t
    weights_untreated = 1 / (1 - pred_t)

    # Calculate the numerator for each observation
    numerator = (t * y * weights_treated) - ((1 - t) * y * weights_untreated)


    # Sum over all observations and divide by the number of observations to get tau_hat
    tau_hat = np.sum(numerator) / len(t)

    return tau_hat

def AIPW(t, pred_t, y, pred_y):
    """
    Calculate the Average Treatment Effect (ATE) using Augmented Inverse Probability of Treatment Weighting (AIPW).

    Parameters:
    - t (array-like): Treatment indicator variable (1 for treated, 0 for untreated).
    - y (array-like): Outcome variable.
    - pred_t (array-like): Predicted propensity score, e(w), for receiving treatment.
    - pred_y (array-like): Predicted outcome variable.

    Returns:
    - tau_hat (float): Estimated treatment effect.
    """

    # Ensure inputs are numpy arrays for element-wise operations
    # t = np.array(t)
    # y = np.array(y)
    # pred_t = np.array(pred_t)
    # pred_y = np.array(pred_y)

    # Calculate the proportion of treated and untreated
    # pt_1 = np.mean(t)
    # pt_0 = 1 - pt_1

    # Calculate stabilized weights without applying trimming
    weights_treated = 1 / pred_t
    weights_untreated = 1 / (1 - pred_t)

    # caculate Y^{a=1}
    y1 = t * (pred_y + (y-pred_y)*weights_treated)
    y0 = (1-t)*(pred_y + (y-pred_y)*weights_untreated)

    # Calculate the AIPW estimate
    tau_hat = np.mean(y1)-np.mean(y0)

    return tau_hat

# stratified standardization estimator
def stratified_standardization(outcome_hat_A1, outcome_hat_A0):
    '''
    Args:
        outcome_hat_A1: prediction of outcome given treatment A1
        outcome_hat_A0: prediction of outcome given treatment A0

    Returns:
        tau_hat: mean of the difference between outcome_hat_A1 and outcome_hat_A0
    '''

    tau = outcome_hat_A1-outcome_hat_A0
    tau_hat = np.mean(tau)

    return tau_hat

'''
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

    #if len(treatment) != len(treatment_hat) or len(outcome) != len(outcome_hat) or len(treatment) != len(outcome):
    #    raise ValueError("All input arrays must have the same length.")

    Y_a = outcome_hat + (treatment / treatment_hat) * (outcome - outcome_hat)
    return np.mean(Y_a)
'''



