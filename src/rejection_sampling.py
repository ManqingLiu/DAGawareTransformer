import pandas as pd
from dataset_u import CausalDataset
from train_u import train
from predict_u import predict
from model import DAGTransformer
from torch.utils.data import DataLoader
import time
import random
import torch
from argparse import ArgumentParser
import json

'''
class RejectionSampler:
    def __init__(self,
                 dag,
                 config,
                 data_train_file,
                 data_holdout_file,
                 model_train_file,
                 model_holdout_file,
                 mask):
        self.model = DAGTransformer(dag=dag, **config['model'])
        self.data_train = pd.read_csv(data_train_file)
        self.data_holdout = pd.read_csv(data_holdout_file)
        self.dag = dag
        self.dataset_train = CausalDataset(self.data_train, self.dag, add_u=True, n_uniform_variables=5, seed_value=0)
        self.dataset_holdout = CausalDataset(self.data_holdout, self.dag, add_u=True, n_uniform_variables=5, seed_value=0)
        self.train_config = config['training']
        self.model_config = config['model']
        self.model_train_file = model_train_file
        self.model_holdout_file = model_holdout_file
        self.mask = mask
        self.total_accepted = 0



    def sample_and_predict_train(self):

        # Generate a new seed and set it before calling CausalDataset
        seed = random.randint(0, 10000)
        dataset_train = CausalDataset(self.data_train, dag, add_u=True, n_uniform_variables=5, seed_value=seed)

        # Create the DataLoader for training data
        batch_size_train = self.train_config['batch_size']
        dataloader_train = DataLoader(dataset_train,
                                           batch_size=batch_size_train,
                                           shuffle=True,
                                           collate_fn=self.dataset_train.collate_fn)
        # Train the model
        train(model=self.model,
              data=self.data_train,
              dataloader=dataloader_train,
              config=self.train_config,
              mask=self.mask,
              model_file=self.model_train_file)


        # Create the DataLoader for holdout data
        batch_size_holdout = self.train_config['batch_size']
        self.dataloader_holdout = DataLoader(self.dataset_holdout,
                                             batch_size=batch_size_holdout,
                                             shuffle=True,
                                             collate_fn=self.dataset_holdout.collate_fn)

        # Make predictions on the holdout data
        predictions_holdout = predict(model=self.model,
                                    data=self.data_holdout,
                                    dag=self.dag,
                                    dataloader=self.dataloader_holdout,
                                    config=train_config,
                                    model_file=self.model_train_file)

        return predictions_holdout

    def sample_and_predict_holdout(self):

        # Generate a new seed and set it before calling CausalDataset
        seed = random.randint(0, 10000)
        dataset_holdout = CausalDataset(self.data_holdout, dag, add_u=True, n_uniform_variables=5, seed_value=seed)

        # Create the DataLoader for holdout data
        batch_size_holdout = self.train_config['batch_size']
        dataloader_holdout = DataLoader(dataset_holdout,
                                             batch_size=batch_size_holdout,
                                             shuffle=True,
                                             collate_fn=self.dataset_holdout.collate_fn)
        # Train the model
        train(model=self.model,
              data=self.data_holdout,
              dataloader=dataloader_holdout,
              config=self.train_config,
              mask=self.mask,
              model_file=self.model_holdout_file)


        # Create the DataLoader for training data
        batch_size_train = self.train_config['batch_size']
        self.dataloader_train = DataLoader(self.dataset_train,
                                           batch_size=batch_size_train,
                                           shuffle=True,
                                           collate_fn=self.dataset_train.collate_fn)

        # Make predictions on the holdout data
        predictions_train = predict(model=self.model,
                                      data=self.data_train,
                                      dag=self.dag,
                                      dataloader=self.dataloader_train,
                                      config=self.train_config,
                                      model_file=self.model_holdout_file)

        return predictions_train


    def compare_and_accept(self, predictions, data):
        # Separate columns in predictions that end with '_prob'
        prob_columns = predictions.filter(regex='_prob$')
        predictions = predictions[predictions.columns.drop(list(prob_columns))]
        # Separate columns in data that starts with U
        u_columns = data.filter(regex='^U')
        data = data[data.columns.drop(list(u_columns))]

        matched_indices = predictions.eq(data).all(axis=1)
        accepted_samples = pd.concat([predictions[matched_indices], prob_columns[matched_indices].reset_index(drop=True)], axis=1)
        self.total_accepted += len(accepted_samples)  # Update the total_accepted variable
        print(f"Accepted {len(accepted_samples)} samples. Total accepted samples: {self.total_accepted}")
        return accepted_samples, accepted_samples.index

    def sample_until(self, desired_samples):
        n = len(self.data_train) + len(self.data_holdout)
        unmatched_rows = set(range(n))  # Initialize the set of unmatched rows
        accepted_samples = []

        while len(accepted_samples) < desired_samples:  # Continue until the desired number of samples is reached
            predictions_train = self.sample_and_predict_train()
            predictions_holdout = self.sample_and_predict_holdout()

            # Compare the predictions with the original datasets
            accepted_train, matched_train_indices = self.compare_and_accept(predictions_train, self.data_train)
            accepted_holdout, matched_holdout_indices = self.compare_and_accept(predictions_holdout, self.data_holdout)

            accepted_samples.append(accepted_train)
            accepted_samples.append(accepted_holdout)
            unmatched_rows -= set(matched_train_indices)
            unmatched_rows -= set(matched_holdout_indices)

            # Check if the desired number of samples is reached
            if len(pd.concat(accepted_samples)) >= desired_samples:
                break

        return pd.concat(accepted_samples)[:desired_samples]  # Return the desired number of samples
'''

class RejectionSampler:
    def __init__(self, dag, config, data_train_file, data_holdout_file, model_file, mask, seed=None):
        self.model = DAGTransformer(dag=dag, **config['model'])
        self.data_train = pd.read_csv(data_train_file)
        self.data_holdout = pd.read_csv(data_holdout_file)
        self.dag = dag
        self.train_config = config['training']
        self.model_file = model_file
        self.mask = mask
        self.total_accepted = 0
        self.seed = seed if seed is not None else random.randint(0, 10000)
        self.initial_train()

    def initial_train(self):
        #seed = random.randint(0, 10000)
        dataset = CausalDataset(pd.concat([self.data_train, self.data_holdout]), self.dag, add_u=True, n_uniform_variables=5, seed_value=self.seed)
        dataloader = DataLoader(dataset, batch_size=self.train_config['batch_size'], shuffle=True, collate_fn=dataset.collate_fn)
        train(model=self.model,
              data=self.data_holdout,
              dataloader=dataloader,
              config=self.train_config,
              mask=self.mask,
              model_file=self.model_file)
        print("Initial training complete.")

    def sample_and_predict(self, data, dataset_class, seed):
        dataset = dataset_class(data, self.dag, add_u=True, n_uniform_variables=5, seed_value=self.seed)
        dataloader = DataLoader(dataset, batch_size=self.train_config['batch_size'], shuffle=True, collate_fn=dataset.collate_fn)
        predictions = predict(
                        model=self.model,
                        data=data,
                        dag=self.dag,
                        dataloader=dataloader,
                        config=self.train_config,
                        model_file=self.model_file)
        return predictions

    '''
    def compare_and_accept(self, predictions, data):
        predictions = predictions.loc[:, ~predictions.columns.str.endswith('_prob')]
        data = data.loc[:, ~data.columns.str.startswith('U')]
        matched_indices = predictions.eq(data).all(axis=1)
        accepted_samples = predictions[matched_indices]
        num_accepted = len(accepted_samples)
        self.total_accepted += num_accepted
        print(f"Accepted {num_accepted} new samples. Total accepted samples: {self.total_accepted}")
        return accepted_samples, matched_indices
    '''

    def compare_and_accept(self, predictions, data):
        # Separate columns in predictions that end with '_prob'
        prob_columns = predictions.filter(regex='_prob$')
        predictions = predictions[predictions.columns.drop(list(prob_columns))]
        # Separate columns in data that starts with U
        u_columns = data.filter(regex='^U')
        data = data[data.columns.drop(list(u_columns))]

        matched_indices = predictions.eq(data).all(axis=1)

        # Create accepted_samples DataFrame
        accepted_samples = pd.concat([data[matched_indices], prob_columns[matched_indices]], axis=1)

        num_accepted = len(accepted_samples)
        self.total_accepted += num_accepted
        print(f"Accepted {num_accepted} new samples. Total accepted samples: {self.total_accepted}")

        return accepted_samples, matched_indices

    def sample_until(self, desired_samples):
        accepted_samples = []
        iteration = 0
        retrain_frequency = 10  # Retrain after every 100 samples accepted

        while len(accepted_samples) < desired_samples:
            predictions_train = self.sample_and_predict(self.data_train, CausalDataset, self.seed)
            predictions_holdout = self.sample_and_predict(self.data_holdout, CausalDataset, self.seed)

            acc_train, _ = self.compare_and_accept(predictions_train, self.data_train)
            acc_holdout, _ = self.compare_and_accept(predictions_holdout, self.data_holdout)

            accepted_samples.extend(acc_train.to_dict('records'))
            accepted_samples.extend(acc_holdout.to_dict('records'))

            if len(accepted_samples) >= desired_samples or iteration % retrain_frequency == 0:
                self.initial_train()  # Retrain the model

            iteration += 1

        return accepted_samples[:desired_samples]




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