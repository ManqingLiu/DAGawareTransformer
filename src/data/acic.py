from argparse import ArgumentParser
import json
import os
import pandas as pd

from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from sklearn.utils import resample



DATA_DIR_NAME = "data"

def load_data_file(file_name, data_dir_name, sep=","):
    file_path = os.path.join(data_dir_name, file_name)
    data = pd.read_csv(file_path, sep=sep)
    return data


def load_acic16(instance=10, raw=True):
    """ Loads single dataset from the 2016 Atlantic Causal Inference Conference data challenge.

    The dataset is based on real covariates but synthetically simulates the treatment assignment
    and potential outcomes. It therefore also contains sufficient ground truth to evaluate
    the effect estimation of causal models.
    The competition introduced 7700 simulated files (100 instances for each of the 77
    data-generating-processes). We provide a smaller sample of one instance from 10
    DGPs. For the full dataset, see the link below to the competition site.

    If used for academic purposes, please consider citing the competition organizers:
     Vincent Dorie, Jennifer Hill, Uri Shalit, Marc Scott, and Dan Cervone. "Automated versus do-it-yourself methods
     for causal inference: Lessons learned from a data analysis competition."
     Statistical Science 34, no. 1 (2019): 43-68.

    Args:
        instance (int): number between 1-10 (inclusive), dataset to load.
        raw (bool): Whether to apply contrast ("dummify") on non-numeric columns
                    If True, returns a (pd.DataFrame, pd.DataFrame) tuple (one for covariates and the second with
                    treatment assignment, noisy potential outcomes and true potential outcomes).

    Returns:
        Bunch: dictionary-like object
               attributes are: `X` (covariates), `a` (treatment assignment), `y` (outcome),
                               `po` (ground truth potential outcomes: `po[0]` potential outcome for controls and
                                `po[1]` potential outcome for treated),
                               `descriptors` (feature description).


    See Also:
        * `Publication <https://projecteuclid.org/euclid.ss/1555056030>`_
        * `Official competition site <http://jenniferhill7.wixsite.com/acic-2016/competition>`_
        * `Official github with data generating code <https://github.com/vdorie/aciccomp/tree/master/2016>`_
    """
    dir_name = os.path.join(DATA_DIR_NAME, "acic")

    X = load_data_file("x.csv", dir_name)
    zymu = load_data_file("zymu_{}.csv".format(instance), dir_name)

    # rename z in zymu to t
    zymu = zymu.rename(columns={"z": "t"})

    # remove all _ in column names of X
    X.columns = X.columns.str.replace("_", "")

    # Create observed outcome y based on treatment assignment a
    zymu["y"] = zymu["t"] * zymu["y1"] + (1 - zymu["t"]) * zymu["y0"]

    # Create ite column which is the difference between mu1 and mu0
    zymu["ite"] = zymu["mu1"] - zymu["mu0"]

    # Identify non-numeric columns
    non_numeric_cols = X.select_dtypes(include=[object]).columns

    for col in non_numeric_cols:
        X[col] = X[col].astype('category').cat.codes

    if raw:
        return X, zymu

def data_split_acic(data, filepaths):
    X = data[0]
    zymu = data[1]
    dataframe = pd.concat([X, zymu], axis=1)

    # split to train and holdout set
    train_data, temp_data = train_test_split(dataframe, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    train_data.to_csv(filepaths["data_train_file"], index=False)
    val_data.to_csv(filepaths["data_val_file"], index=False)
    test_data.to_csv(filepaths["data_test_file"], index=False)

    return train_data, val_data, test_data

def process_all_samples(config_dir):
    for instance in range(1, 11):
        config_file = os.path.join(config_dir, f"acic_sample{instance}.json")
        with open(config_file) as f:
            config = json.load(f)

        filepaths = config["filepaths"]
        data = load_acic16(instance=instance, raw=True)
        data_split_acic(data, filepaths)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config_dir", type=str, required=True,
                        help="Directory containing the JSON config files.")
    args = parser.parse_args()

    process_all_samples(args.config_dir)

