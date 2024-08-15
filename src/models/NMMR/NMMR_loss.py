from src.models.NMMR.kernel_utils import calculate_kernel_matrix_batched
import torch
from torch.nn import CrossEntropyLoss


def NMMR_loss_transformer(model_output,
                          target,
                          treatment,
                          kernel_matrix,
                          loss_name: str,
                          alpha = 0.01,
                          return_items: bool = False):
    residual = target - model_output
    n = residual.shape[0]
    K = kernel_matrix

    if loss_name == "U_statistic":
        # Calculate U statistic (see Serfling 1980)
        K.fill_diagonal_(0)
        loss = (residual.T @ K @ residual) / (n * (n - 1))
    elif loss_name == "V_statistic":
        # Calculate V statistic (see Serfling 1980)
        loss = (residual.T @ K @ residual) / (n ** 2)
    else:
        raise ValueError(f"{loss_name} is not valid. Must be 'U_statistic' or 'V_statistic'.")

    # Contrastive Loss Component #
    treatment_mask = (treatment['treatment'].unsqueeze(0) != treatment['treatment'].unsqueeze(1))
    
    # Compute pairwise differences
    differences = model_output.unsqueeze(0) - model_output.unsqueeze(1)
    masked_differences = differences.squeeze() * treatment_mask
    contrastive_loss = torch.negative(torch.norm(differences))

    # You can weight this contrastive loss with a hyperparameter
    total_loss = loss[0, 0] + alpha * contrastive_loss

    if return_items:
        batch_items = {
            'residual': residual,
            'kernel_matrix': K,
            'contrastive_loss': contrastive_loss.item(),
            'causal_loss': loss[0, 0].item(),
            'total_loss': total_loss.item()
        }
        return total_loss, batch_items
    else:
        return total_loss

def NMMR_loss(model_output, target, kernel_matrix, loss_name: str):  # batch_indices=None):
    residual = target - model_output
    n = residual.shape[0]
    K = kernel_matrix

    if loss_name == "U_statistic":
        # calculate U statistic (see Serfling 1980)
        K.fill_diagonal_(0)
        loss = (residual.T @ K @ residual) / (n * (n-1))
    elif loss_name == "V_statistic":
        # calculate V statistic (see Serfling 1980)
        loss = (residual.T @ K @ residual) / (n ** 2)
    else:
        raise ValueError(f"{loss_name} is not valid. Must be 'U_statistic' or 'V_statistic'.")

    return loss[0, 0]


def NMMR_loss_batched(model_output, target, kernel_inputs, kernel, batch_size: int, loss_name: str):
    residual = target - model_output
    n = residual.shape[0]

    loss = 0
    for i in range(0, n, batch_size):
        partial_kernel_matrix = calculate_kernel_matrix_batched(kernel_inputs, (i, i+batch_size), kernel)
        if loss_name == "V_statistic":
            factor = n ** 2
        if loss_name == "U_statistic":
            factor = n * (n-1)
            # zero out the main diagonal of the full matrix
            for row_idx in range(partial_kernel_matrix.shape[0]):
                partial_kernel_matrix[row_idx, row_idx+i] = 0
        temp_loss = residual[i:(i+batch_size)].T @ partial_kernel_matrix @ residual / factor
        loss += temp_loss[0, 0]
    return loss
