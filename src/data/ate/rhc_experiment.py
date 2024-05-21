import pathlib
import os.path as op
import numpy as np
import pandas as pd

from src.data.ate.data_class import PVTrainDataSet, RHCTestDataSet

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.parent.joinpath("data/right_heart_catheterization")


def generate_train_rhc(use_all_X: bool) -> PVTrainDataSet:
    # load train split
    train_split = pd.read_csv(op.join(DATA_PATH, "rhc_train.csv"))

    # load X feature list
    if use_all_X:
        X_names = pd.read_csv(op.join(DATA_PATH, "RHC_X_allfeatures_list.csv"))
    else:
        X_names = pd.read_csv(op.join(DATA_PATH, "RHC_X_significantfeatures_list.csv"))

    # subset the train split to X_names
    train_X = train_split[X_names.variable.tolist()].to_numpy()

    # separate the key proximal variables
    rhc_treatment = np.expand_dims(train_split['swang1'].to_numpy(), axis=-1)
    survival_time = np.expand_dims(train_split['t3d30'].to_numpy(), axis=-1)
    Z_pafi = np.expand_dims(train_split['pafi1'].to_numpy(), axis=-1)
    Z_paco21 = np.expand_dims(train_split['paco21'].to_numpy(), axis=-1)
    W_ph1 = np.expand_dims(train_split['ph1'].to_numpy(), axis=-1)
    W_hema1 = np.expand_dims(train_split['hema1'].to_numpy(), axis=-1)

    return PVTrainDataSet(treatment=rhc_treatment,
                          treatment_proxy=np.c_[Z_pafi, Z_paco21],
                          outcome_proxy=np.c_[W_ph1, W_hema1],
                          outcome=survival_time,
                          backdoor=train_X)


def generate_val_rhc(use_all_X: bool) -> PVTrainDataSet:
    # load the validation split
    val_split = pd.read_csv(op.join(DATA_PATH, "rhc_val.csv"))

    # load X feature list
    if use_all_X:
        X_names = pd.read_csv(op.join(DATA_PATH, "RHC_X_allfeatures_list.csv"))
    else:
        X_names = pd.read_csv(op.join(DATA_PATH, "RHC_X_significantfeatures_list.csv"))

    # subset the validation split to X_names
    val_X = val_split[X_names.variable.tolist()].to_numpy()

    # separate the key proximal variables
    rhc_treatment = np.expand_dims(val_split['swang1'].to_numpy(), axis=-1)
    survival_time = np.expand_dims(val_split['t3d30'].to_numpy(), axis=-1)
    Z_pafi = np.expand_dims(val_split['pafi1'].to_numpy(), axis=-1)
    Z_paco21 = np.expand_dims(val_split['paco21'].to_numpy(), axis=-1)
    W_ph1 = np.expand_dims(val_split['ph1'].to_numpy(), axis=-1)
    W_hema1 = np.expand_dims(val_split['hema1'].to_numpy(), axis=-1)

    return PVTrainDataSet(treatment=rhc_treatment,
                          treatment_proxy=np.c_[Z_pafi, Z_paco21],
                          outcome_proxy=np.c_[W_ph1, W_hema1],
                          outcome=survival_time,
                          backdoor=val_X)


def generate_test_rhc(use_all_X: bool) -> RHCTestDataSet:
    """
    Generates the data required to compute the Average Treatment Effect (ATE) for right-heart catheterization (RHC)
    on survival time (up to 30 days).

    Parameters
    ----------
    use_all_X: True/False to indicate whether to use all the measured confounders as X, or just the statistically
    significant subset determined during preprocessing.

    Returns
    -------
    A named tuple with 3 keys: `treatment`, `outcome_proxy`, and `backdoor`. `treatment` is the array [0, 1] because the
    treatment variable is binary and we are interested in the ATE. In contrast, `outcome_proxy` has a shape of (574, 2),
    which is 10% of the patients held out from training and validation. `backdoor` will have a shape of (574, 53) if
    `use_all_X` is True, and (574, 22) otherwise.
    """

    # load the test split
    test_split = pd.read_csv(op.join(DATA_PATH, "rhc_test.csv"))

    # load X feature list
    if use_all_X:
        X_names = pd.read_csv(op.join(DATA_PATH, "RHC_X_allfeatures_list.csv"))
    else:
        X_names = pd.read_csv(op.join(DATA_PATH, "RHC_X_significantfeatures_list.csv"))

    # subset the test split to X_names
    test_X = test_split[X_names.variable.tolist()].to_numpy()

    # separate the outcome proxy variables
    W_ph1 = np.expand_dims(test_split['ph1'].to_numpy(), axis=-1)
    W_hema1 = np.expand_dims(test_split['hema1'].to_numpy(), axis=-1)

    return RHCTestDataSet(treatment=np.array([[0], [1]]),
                          outcome_proxy=np.c_[W_ph1, W_hema1],
                          backdoor=test_X)
