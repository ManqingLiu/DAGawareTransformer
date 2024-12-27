from argparse import ArgumentParser
import json
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def data_split_lalonde(data, dag):
    # rename re78 as y
    data = data.rename(columns={"re78": "y"})
    # rename treatment as t
    data = data.rename(columns={"treat": "t"})

    data = data[dag["nodes"]]

    # split to train and holdout set
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    train_data.to_csv(filepaths["data_train_file"], index=False)
    val_data.to_csv(filepaths["data_val_file"], index=False)
    test_data.to_csv(filepaths["data_test_file"], index=False)

    return train_data, val_data, test_data


def data_sample_split_lalonde(data: pd.DataFrame, dag: dict, dataname: str, iters: int = 50,
                              output_dir: str = 'data/lalonde') -> None:
    # Rename columns
    data = data.rename(columns={"re78": "y", "treat": "t"})
    data = data[dag["nodes"]]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for i in range(iters):
        # Resample the data with replacement
        resampled_data = resample(data, n_samples=len(data), random_state=i)

        # Split to train and holdout set
        train_data, temp_data = train_test_split(resampled_data, test_size=0.3, random_state=i)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=i)

        # Create subdirectory for this sample if it doesn't exist
        sample_dir = os.path.join(output_dir, f'{dataname}/sample{i}')
        os.makedirs(sample_dir, exist_ok=True)

        # Save the splits for this iteration
        train_data.to_csv(os.path.join(sample_dir, f'train_data_{i}.csv'), index=False)
        val_data.to_csv(os.path.join(sample_dir, f'val_data_{i}.csv'), index=False)
        test_data.to_csv(os.path.join(sample_dir, f'test_data_{i}.csv'), index=False)




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    filepaths = config["filepaths"]

    with open(filepaths["dag"]) as f:
        dag = json.load(f)

    data = pd.read_csv(filepaths["data_file"])

    data_sample_split_lalonde(data, dag, dataname='ldw_cps', iters=10, output_dir='data/lalonde')