from argparse import ArgumentParser
import json
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

from src.data.ate.data_class import PVTrainDataSet
from src.data.ate import (
    generate_train_data_ate,
    generate_val_data_ate,
    generate_test_data_ate,
)
from src.data.ate.data_class import (
    PVTrainDataSet,
    PVTrainDataSetTorch,
    PVTestDataSetTorch,
)


class CausalDataset(Dataset):
    def __init__(self, data, dag, random_seed: int):
        self.data = data
        self.random_seed = random_seed

        if isinstance(self.data, pd.DataFrame):
            self.data_binned = self.data.copy()
        elif isinstance(self.data, dict):
            self.data_binned = {k: None for k, _ in self.data.items()}

        self.dag = dag
        self.bin_edges = {}

        self.num_nodes = len(self.dag["nodes"])
        self.dag["node_ids"] = dict(zip(self.dag["nodes"], range(self.num_nodes)))

        if isinstance(self.data, pd.DataFrame):
            self.bin_columns()
        elif isinstance(self.data, dict):
            self.bin_columns_for_ndarray()

    def get_labels(self):
        """Returns the treatment column to be used by torchsampler's ImbalancedDatasetSampler class 
        so that it can oversample the treated group and balance untreated/treated observations"""
        return self.data["t"].values
    
    def __len__(self):
        if isinstance(self.data, pd.DataFrame):
            return len(self.data)
        elif isinstance(self.data, dict):
            first_key = next(iter(self.data))
            return len(self.data[first_key])

    def __getitem__(self, idx):
        if isinstance(self.data, pd.DataFrame):
            return self.data.iloc[idx], self.data_binned.iloc[idx]
        elif isinstance(self.data, dict):
            return {key: self.data[key][idx] for key in self.data}, {key: self.data_binned[key][idx] for key in self.data_binned}
    
    def collate_fn(self, batch_list):
        batch_data = {key: [] for key in self.dag["input_nodes"]}
        batch_binned = {key: [] for key in self.dag["input_nodes"]}

        for data, binned in batch_list:
            for key in self.dag["input_nodes"]:
                batch_data[key].append(torch.tensor(data[key]))
                batch_binned[key].append(torch.tensor(binned[key]))

        collated_data = {key: torch.stack(batch_data[key]) for key in batch_data}
        collated_binned = {key: torch.stack(batch_binned[key]) for key in batch_binned}

        return collated_data, collated_binned

    def bin_columns(self):
        for column, params in self.dag["input_nodes"].items():
            num_bins = params["num_categories"]
            binner = KBinsDiscretizer(
                n_bins=num_bins, encode="ordinal", strategy="uniform", subsample=None, random_state=self.random_seed
            )
            if num_bins > 2:
                self.data_binned[column] = binner.fit_transform(
                    self.data[column].values.reshape(-1, 1)
                ).flatten()
                self.data_binned[column] = self.data_binned[column].astype(int)
                self.bin_edges[column] = binner.bin_edges_[0]
            elif num_bins == 2:
                self.data_binned[column] = pd.cut(
                    self.data[column], bins=2, labels=False
                )

    def bin_columns_for_ndarray(self):
        # self.data arrays w/ shape (n, 1) -> self.data_binned arrays w/ shape (n,)
        for column, params in self.dag["input_nodes"].items():
            num_bins = params["num_categories"]
            binner = KBinsDiscretizer(
                n_bins=num_bins, encode="ordinal", strategy="uniform", subsample=None, random_state=self.random_seed
            )
            if num_bins > 2:
                self.data_binned[column] = binner.fit_transform(
                    self.data[column].reshape(-1, 1)
                ).flatten()
                self.data_binned[column] = self.data_binned[column].astype(int)
                self.bin_edges[column] = binner.bin_edges_[0]
            elif num_bins == 2:
                self.data_binned[column] = pd.cut(
                    self.data[column], bins=2, labels=False
                ).to_numpy()

    def get_bin_left_edges(self):
        return {k: v[:-1] for k, v in self.bin_edges.items()}


def make_train_data(
    data_config: Dict[str, Any], dag: Dict[str, Any], random_seed: int
) -> CausalDataset:
    """ Returns a CausalDataset where self.data has 5 keys each with shape (num_samples, ) """
    train_data = generate_train_data_ate(data_config=data_config, rand_seed=random_seed)
    train_t = PVTrainDataSetTorch.from_numpy(train_data)

    train_data_dict = {
        "treatment": train_t.treatment,
        "treatment_proxy1": train_t.treatment_proxy[:, 0].unsqueeze(1),
        "treatment_proxy2": train_t.treatment_proxy[:, 1].unsqueeze(1),
        "outcome_proxy": train_t.outcome_proxy,
        "outcome": train_t.outcome,
    }

    return CausalDataset(train_data_dict, dag, random_seed)


def make_validation_data(
    data_config: Dict[str, Any], dag: Dict[str, Any], random_seed: int
) -> CausalDataset:
    """ Returns a CausalDataset where self.data has 5 keys each with shape (num_samples, ) """
    val_data = generate_val_data_ate(data_config=data_config, rand_seed=random_seed + 1)
    val_data_t = PVTrainDataSetTorch.from_numpy(val_data)

    val_data_dict = {
        "treatment": val_data_t.treatment,
        "treatment_proxy1": val_data_t.treatment_proxy[:, 0].unsqueeze(1),
        "treatment_proxy2": val_data_t.treatment_proxy[:, 1].unsqueeze(1),
        "outcome_proxy": val_data_t.outcome_proxy,
        "outcome": val_data_t.outcome,
    }

    return CausalDataset(val_data_dict, dag, random_seed)


def make_test_data(
    data_config: Dict[str, Any], val_data: CausalDataset, dag: Dict[str, Any]
) -> CausalDataset:
    
    '''num_W_test is the numnber of samples in the validation set and the number of samples
    used to estimate the expected value of the bridge function at each intervention level.
    
    Returns a CausalDataset where self.data has 5 keys each with shape (num intervention levels, num val samples)'''

    test_data = generate_test_data_ate(data_config=data_config)
    test_data_t = PVTestDataSetTorch.from_numpy(test_data)

    intervention_array_len = test_data_t.treatment.shape[0]
    num_W_test = val_data.data["outcome_proxy"].shape[0]
    treatment = test_data_t.treatment.expand(-1, num_W_test)  # (intervention_array_len, num_W_test)

    # Z1, Z2, W and Y initially have shape (num_W_test,) but are copied column-wise to have shape (num_W_test, intervention_array_len)
    treatment_proxy1 = val_data.data['treatment_proxy1'].expand(-1, intervention_array_len)
    treatment_proxy2 = val_data.data['treatment_proxy2'].expand(-1, intervention_array_len)
    outcome_proxy = val_data.data['outcome_proxy'].expand(-1, intervention_array_len)
    outcome = val_data.data['outcome'].expand(-1, intervention_array_len)

    test_data_dict = {
        "treatment": treatment.reshape(-1, 1),
        "treatment_proxy1": treatment_proxy1.T.reshape(-1, 1),
        "treatment_proxy2": treatment_proxy2.T.reshape(-1, 1),
        "outcome_proxy": outcome_proxy.T.reshape(-1, 1),
        "outcome": outcome.T.reshape(-1, 1),
    }

    return CausalDataset(test_data_dict, dag, random_seed=data_config['random_seed']), test_data.structural


class PredictionTransformer:
    def __init__(self, bin_edges):
        self.bin_edges = bin_edges
        self.bin_midpoints = {k: (v[1:] + v[:-1]) / 2 for k, v in bin_edges.items()}

    def transform(self, predictions):
        t_predictions = predictions[:, :2]
        y_predictions = predictions[:, 2:]

        t_prob = t_predictions[:, 1]  # Probability of t=1
        # Calculate y_expected_value based on the number of columns in y_predictions
        if y_predictions.shape[1] == 2:
            y_expected_value = y_predictions[:, 1]
        elif y_predictions.shape[1] > 2:
            y_expected_value = np.sum(y_predictions * self.bin_midpoints["y"], axis=1)

        transformed_predictions = pd.DataFrame(
            {"t_prob": t_prob, "pred_y": y_expected_value}
        )

        return transformed_predictions

    def transform_proximal(self, predictions, n_sample):
        # To Do: check the shape of predictions
        expected_value = np.sum(predictions * self.bin_midpoints["outcome"], axis=1)

        # Transform the expected_value into a tensor
        transformed_predictions = torch.tensor(
            expected_value, dtype=torch.float32
        ).view(10, n_sample, 1)

        return transformed_predictions

    def transform_frontdoor(self, predictions):
        # m_predictions is the 3rd and 4th columns of the predictions
        m_predictions = predictions[:, 2:4]
        y_predictions = predictions[:, 4:]

        m_prob = m_predictions[:, 1]  # Probability of m=1
        if y_predictions.shape[1] == 2:
            y_expected_value = y_predictions[:, 1]
        elif y_predictions.shape[1] > 2:
            y_expected_value = np.sum(y_predictions * self.bin_midpoints["Y"], axis=1)

        transformed_predictions = pd.DataFrame(
            {"m_prob": m_prob, "pred_y": y_expected_value}
        )

        return transformed_predictions


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dag", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--train_output_file", type=str, required=True)
    parser.add_argument("--val_output_file", type=str, required=True)
    parser.add_argument("--test_output_file", type=str, required=True)

    args = parser.parse_args()
    data = pd.read_csv(args.data_file)

    # cleaning
    # rename re78 as y
    data = data.rename(columns={"re78": "y"})
    # rename treatment as t
    data = data.rename(columns={"treat": "t"})

    with open(args.dag) as f:
        print(f"Loading dag file from {args.dag}")
        dag = json.load(f)

    data = data[dag["nodes"]]
    # split to train and holdout set
    train_data, temp_data = train_test_split(data, test_size=0.7, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    train_data.to_csv(args.train_output_file, index=False)
    val_data.to_csv(args.val_output_file, index=False)
    test_data.to_csv(args.test_output_file, index=False)

    # print average t in train_data
    print(f"Average t in train data: {train_data['t'].mean()}")
    # print average t in val_data
    print(f"Average t in val data: {val_data['t'].mean()}")
