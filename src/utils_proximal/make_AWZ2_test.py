import torch
from src.data.ate.data_class import PVTrainDataSetTorch, PVTestDataSetTorch


def make_AWZ2_test(test_data_t: PVTestDataSetTorch,
                   val_data_t: PVTrainDataSetTorch) -> torch.Tensor:
    """
    Creates a 3-dim Tensor with shape (intervention_array_len, n_samples, 14)
    This will contain the test values for do(A) chosen by Xu et al. as well as `n_samples` random draws for W & Z
    The features will be arranged in the following order:

    [A, W, (Z_1, Z_2),
    A^2, W^2, (Z_1^2, Z_2^2),
    A*W, (A*Z_1, A*Z_2), (W*Z_1, W*Z_2), Z_1*Z_2]

    This is why the final axis has a size of 14.
    This includes all second order terms of (A, W, Z) and their cross-products.

    Parameters
    ----------
    test_data_t: expected to have a 'treatment' vector.
    val_data_t: expected to have 'outcome_proxy' and 'treatment_proxy' vectors.

    Returns
    -------
    AWZ2_test: 3-dim torch.Tensor with shape (intervention_array_len, n_samples, 14)

    """
    # get array sizes
    intervention_array_len = test_data_t.treatment.shape[0]
    num_val_samples = val_data_t.treatment.shape[0]

    # tile A, W and Z to create a tensor with shape (intervention_array_len, n_samples, 14)
    AWZ2_test = torch.stack((test_data_t.treatment.expand(-1, num_val_samples),
                             val_data_t.outcome_proxy.expand(-1, intervention_array_len).T,
                             val_data_t.treatment_proxy[:, 0:1].expand(-1, intervention_array_len).T,
                             val_data_t.treatment_proxy[:, 1:2].expand(-1, intervention_array_len).T,
                             (test_data_t.treatment ** 2).expand(-1, num_val_samples),
                             (val_data_t.outcome_proxy ** 2).expand(-1, intervention_array_len).T,
                             (val_data_t.treatment_proxy[:, 0:1] ** 2).expand(-1, intervention_array_len).T,
                             (val_data_t.treatment_proxy[:, 1:2] ** 2).expand(-1, intervention_array_len).T,
                             test_data_t.treatment.expand(-1, num_val_samples) * val_data_t.outcome_proxy.expand(-1, intervention_array_len).T,
                             test_data_t.treatment.expand(-1, num_val_samples) * val_data_t.treatment_proxy[:, 0:1].expand(-1, intervention_array_len).T,
                             test_data_t.treatment.expand(-1, num_val_samples) * val_data_t.treatment_proxy[:, 1:2].expand(-1, intervention_array_len).T,
                             (val_data_t.outcome_proxy * val_data_t.treatment_proxy[:, 0:1]).expand(-1, intervention_array_len).T,
                             (val_data_t.outcome_proxy * val_data_t.treatment_proxy[:, 1:2]).expand(-1, intervention_array_len).T,
                             (val_data_t.treatment_proxy[:, 0:1] * val_data_t.treatment_proxy[:, 1:2]).expand(-1, intervention_array_len).T),
                            dim=-1)

    return AWZ2_test
