import pandas as pd
from dataset_u import CausalDataset
from train_u import train
from predict_u import predict
from model import DAGTransformer
from torch.utils.data import DataLoader
import time
import random
import numpy as np
import torch
from argparse import ArgumentParser
import json


class RejectionSampler:
    def __init__(self, dag, config, data_train_file, data_holdout_file, model_file, mask,n_uniform_variables, seed=None):
        self.model = DAGTransformer(dag=dag, **config['model'])
        self.data_train = pd.read_csv(data_train_file)
        self.data_holdout = pd.read_csv(data_holdout_file)
        self.dag = dag
        self.train_config = config['training']
        self.model_file = model_file
        self.mask = mask
        self.n_uniform_variables = n_uniform_variables
        self.total_accepted = 0
        self.seed = seed if seed is not None else random.randint(0, 10000)
        self.initial_train()
        self.remaining_indices_train = self.data_train.index.copy()
        self.remaining_indices_holdout = self.data_holdout.index.copy()

    def initial_train(self):
        # Combine and shuffle data initially
        combined_data = pd.concat([self.data_train, self.data_holdout], ignore_index=True)
        dataset = CausalDataset(combined_data, self.dag, add_u=True, n_uniform_variables=self.n_uniform_variables, seed_value=self.seed)
        dataloader = DataLoader(dataset, batch_size=self.train_config['batch_size'], shuffle=True, collate_fn=dataset.collate_fn)
        train(model=self.model, data=combined_data, dataloader=dataloader, config=self.train_config, mask=self.mask, model_file=self.model_file)
        print("Initial training complete.")

    def sample_and_predict(self, data, remaining_indices, dataset_class, seed):
        data_sample = data.loc[remaining_indices]
        dataset = dataset_class(data_sample, self.dag, add_u=True, n_uniform_variables=self.n_uniform_variables, seed_value=seed)
        dataloader = DataLoader(dataset, batch_size=len(data_sample), shuffle=True, collate_fn=dataset.collate_fn)
        predictions = predict(model=self.model, data=data_sample, dag=self.dag, dataloader=dataloader, model_file=self.model_file)
        return predictions, data_sample

    def compare_and_accept(self, predictions, data):
        prob_columns = predictions.filter(regex='_prob$')
        predictions = predictions.drop(columns=prob_columns.columns)
        u_columns = data.filter(regex='^U')
        data = data.drop(columns=u_columns.columns)

        matched_indices = predictions.eq(data).all(axis=1)
        accepted_samples = pd.concat([data[matched_indices], prob_columns[matched_indices]], axis=1)

        num_accepted = len(accepted_samples)
        self.total_accepted += num_accepted
        print(f"Accepted {num_accepted} new samples. Total accepted samples: {self.total_accepted}")

        return accepted_samples, matched_indices

    def sample_until(self, desired_samples):
        accepted_samples = pd.DataFrame()

        while len(accepted_samples) < desired_samples:
            if not self.remaining_indices_train.empty:
                predictions_train, sample_train = self.sample_and_predict(self.data_train, self.remaining_indices_train, CausalDataset, self.seed)
                acc_train, matched_indices_train = self.compare_and_accept(predictions_train, sample_train)
                accepted_samples = pd.concat([accepted_samples, acc_train])
                self.remaining_indices_train = self.remaining_indices_train.difference(matched_indices_train.index)

            if not self.remaining_indices_holdout.empty:
                predictions_holdout, sample_holdout = self.sample_and_predict(self.data_holdout, self.remaining_indices_holdout, CausalDataset, self.seed)
                acc_holdout, matched_indices_holdout = self.compare_and_accept(predictions_holdout, sample_holdout)
                accepted_samples = pd.concat([accepted_samples, acc_holdout])
                self.remaining_indices_holdout = self.remaining_indices_holdout.difference(matched_indices_holdout.index)

            if len(accepted_samples) >= desired_samples:
                break

        return accepted_samples.iloc[:desired_samples].to_dict('records')


'''
class RejectionSampler:
    def __init__(self, dag, config, data_train_file, data_holdout_file, model_file, mask, n_uniform_variables,
                 seed=None):
        self.model = DAGTransformer(dag=dag, **config['model'])
        self.data_train = pd.read_csv(data_train_file)
        self.data_holdout = pd.read_csv(data_holdout_file)
        self.dag = dag
        self.train_config = config['training']
        self.model_file = model_file
        self.mask = mask
        self.n_uniform_variables = n_uniform_variables
        self.total_accepted = 0
        self.seed = seed if seed is not None else random.randint(0, 10000)
        self.u_success = []  # List to track successful u values
        self.initial_train()
        self.remaining_indices_train = self.data_train.index.copy()
        self.remaining_indices_holdout = self.data_holdout.index.copy()

    def initial_train(self):
        # Initial training process remains the same
        combined_data = pd.concat([self.data_train, self.data_holdout], ignore_index=True)
        dataset = CausalDataset(combined_data, self.dag, add_u=True, n_uniform_variables=self.n_uniform_variables,
                                seed_value=self.seed)
        dataloader = DataLoader(dataset, batch_size=self.train_config['batch_size'], shuffle=True,
                                collate_fn=dataset.collate_fn)
        train(model=self.model, data=combined_data, dataloader=dataloader, config=self.train_config, mask=self.mask,
              model_file=self.model_file)
        print("Initial training complete.")

    def sample_and_predict(self, data, remaining_indices, dataset_class, seed):
        # Adjust the way u is sampled based on past successes
        mean_u = np.mean(self.u_success, axis=0) if self.u_success else np.zeros(self.n_uniform_variables)
        std_u = np.std(self.u_success, axis=0) if len(self.u_success) > 1 else np.ones(self.n_uniform_variables)

        # Modify the dataset class to accept new u distribution parameters
        data_sample = data.loc[remaining_indices]
        dataset = dataset_class(data_sample, self.dag, add_u=True, n_uniform_variables=self.n_uniform_variables,
                                seed_value=seed, mean_u=mean_u, std_u=std_u)
        dataloader = DataLoader(dataset, batch_size=len(data_sample), shuffle=True, collate_fn=dataset.collate_fn)
        predictions = predict(model=self.model, data=data_sample, dag=self.dag, dataloader=dataloader,
                              model_file=self.model_file)
        return predictions, data_sample

    def compare_and_accept(self, predictions, data):
        # This method remains unchanged
        prob_columns = predictions.filter(regex='_prob$')
        predictions = predictions.drop(columns=prob_columns.columns)
        u_columns = data.filter(regex='^U')
        data = data.drop(columns=u_columns.columns)

        matched_indices = predictions.eq(data).all(axis=1)
        accepted_samples = pd.concat([data[matched_indices], prob_columns[matched_indices]], axis=1)
        self.u_success.extend(u_columns[matched_indices].values)  # Store successful u values

        num_accepted = len(accepted_samples)
        self.total_accepted += num_accepted
        print(f"Accepted {num_accepted} new samples. Total accepted samples: {self.total_accepted}")

        return accepted_samples, matched_indices

    def sample_until(self, desired_samples):
        accepted_samples = pd.DataFrame()

        while len(accepted_samples) < desired_samples:
            if not self.remaining_indices_train.empty:
                predictions_train, sample_train = self.sample_and_predict(self.data_train, self.remaining_indices_train,
                                                                          CausalDataset, self.seed)
                acc_train, matched_indices_train = self.compare_and_accept(predictions_train, sample_train)
                accepted_samples = pd.concat([accepted_samples, acc_train])
                self.remaining_indices_train = self.remaining_indices_train.difference(matched_indices_train.index)

            if not self.remaining_indices_holdout.empty:
                predictions_holdout, sample_holdout = self.sample_and_predict(self.data_holdout,
                                                                              self.remaining_indices_holdout,
                                                                              CausalDataset, self.seed)
                acc_holdout, matched_indices_holdout = self.compare_and_accept(predictions_holdout, sample_holdout)
                accepted_samples = pd.concat([accepted_samples, acc_holdout])
                self.remaining_indices_holdout = self.remaining_indices_holdout.difference(
                    matched_indices_holdout.index)

            if len(accepted_samples) >= desired_samples:
                break

        return accepted_samples.iloc[:desired_samples].to_dict('records')
'''



if __name__ == "__main__":
    from argparse import ArgumentParser
    import json
    import pandas as pd

    parser = ArgumentParser()
    parser.add_argument('--dag', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_train_file', type=str, required=True)
    parser.add_argument('--data_holdout_file', type=str, required=True)
    parser.add_argument('--model_train_file', type=str, required=True)
    parser.add_argument('--model_holdout_file', type=str, required=True)

    args = parser.parse_args()

    with open(args.dag) as f:
        print(f'Loading dag file from {args.dag}')
        dag = json.load(f)

    with open(args.config) as f:
        print(f'Loading config file from {args.config}')
        config = json.load(f)

    train_config = config['training']
    model_config = config['model']

    # Move all of this to a utils function for load_dag and load_data
    num_nodes = len(dag['nodes'])
    dag['node_ids'] = dict(zip(dag['nodes'], range(num_nodes)))
    start_time = time.time()
    sampler = RejectionSampler(dag,
                               config,
                               args.data_train_file,
                               args.data_holdout_file,
                               args.model_train_file,
                               #args.model_holdout_file,
                               mask=True,
                               n_uniform_variables=1,
                               seed=random.randint(0, 10000))
    original_sample_size = len(sampler.data_train) + len(sampler.data_holdout)
    accepted_samples = sampler.sample_until(desired_samples=100)
    end_time = time.time()
    # Calculate and print the total wall time
    total_wall_time = end_time - start_time
    # Convert the total wall time to minutes and seconds
    minutes, seconds = divmod(total_wall_time, 60)
    print(f"Total wall time used: {minutes} minutes and {seconds} seconds")

    print(f"Number of accepted samples: {len(accepted_samples)}")

    # Save the accepted_samples as a CSV file
    accepted_samples_df = pd.DataFrame(accepted_samples)
    accepted_samples_df.to_csv('data/unmeasured_confounding/accepted_samples_n100.csv', index=False)