import torch
from src.data.ate.data_class import PVTrainDataSetTorch, PVTestDataSetTorch


def make_AWZ_test(test_data_t: PVTestDataSetTorch,
                  val_data_t: PVTrainDataSetTorch) -> torch.Tensor:
    """
    Creates a 3-dim Tensor with shape (intervention_array_len, n_samples, 4)
    This will contain the test values for do(A) chosen by Xu et al. as well as len(val_data_t) random draws for W & Z

    Note: Z is 2-dimensional, hence the 4 in the last dimension

    Parameters
    ----------
    test_data_t: expected to have a 'treatment' vector.
    val_data_t: expected to have 'outcome_proxy' and 'treatment_proxy' vectors.

    Returns
    -------
    AWZ_test: 3-dim torch.Tensor with shape (intervention_array_len, n_samples, 4)

    """
    # get array sizes
    intervention_array_len = test_data_t.treatment.shape[0]
    num_val_samples = val_data_t.treatment.shape[0]

    # tile A, W and Z to create a tensor with shape (intervention_array_len, n_samples, 4)
    # '4' because A and W are 1-dim, but Z is 2-dim
    temp_A = test_data_t.treatment.expand(-1, num_val_samples)
    temp_W = val_data_t.outcome_proxy.expand(-1, intervention_array_len)
    temp_AW = torch.stack((temp_A, temp_W.T), dim=-1)
    temp_Z = torch.unsqueeze(val_data_t.treatment_proxy, dim=-1).repeat(1, 1, intervention_array_len)
    temp_Z = torch.transpose(temp_Z.T, dim0=1, dim1=2)
    AWZ_test = torch.cat((temp_AW, temp_Z), dim=-1)

    return AWZ_test

def make_AW_test(test_data_t: PVTestDataSetTorch,
                  val_data_t: PVTrainDataSetTorch) -> torch.Tensor:
    """
    Creates a 2-dim Tensor with shape (intervention_array_len, n_samples, 4)
    This will contain the test values for do(A) chosen by Xu et al. as well as len(val_data_t) random draws for W

    Parameters
    ----------
    test_data_t: expected to have a 'treatment' vector.
    val_data_t: expected to have 'outcome_proxy' and 'treatment_proxy' vectors.

    Returns
    -------
    AW_test: 2-dim torch.Tensor with shape (intervention_array_len, n_samples, 4)

    """
    # get array sizes
    intervention_array_len = test_data_t.treatment.shape[0]
    num_val_samples = val_data_t.treatment.shape[0]

    # tile A, W and Z to create a tensor with shape (intervention_array_len, n_samples, 4)
    # '4' because A and W are 1-dim, but Z is 2-dim
    temp_A = test_data_t.treatment.expand(-1, num_val_samples)
    temp_W = val_data_t.outcome_proxy.expand(-1, intervention_array_len)
    temp_AW = torch.stack((temp_A, temp_W.T), dim=-1)

    return temp_AW