import numpy as np
import torch


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, length_scale=1):
    return torch.exp(-1. * torch.sum((x - y) ** 2, axis=0) / (2 * (length_scale ** 2)))


def calculate_kernel_matrix(dataset, kernel=rbf_kernel, **kwargs):
    tensor = dataset.permute(1, 0)
    tensor1 = tensor.unsqueeze(dim=2)
    tensor2 = tensor.unsqueeze(dim=1)

    return kernel(tensor1, tensor2, **kwargs)

def calculate_kernel_matrix_batched(dataset, batch_indices:tuple, kernel=rbf_kernel, **kwargs):
    #calculate_kernel_matrix_batched(kernel_inputs, (i, i+batch_size), kernel)
    tensor = dataset.permute(1,0)
    tensor1 = tensor.unsqueeze(dim=2)
    tensor1 = tensor1[:, batch_indices[0]:batch_indices[1], :]
    tensor2 = tensor.unsqueeze(dim=1)

    return kernel(tensor1, tensor2, **kwargs)