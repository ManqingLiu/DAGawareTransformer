import pandas as pd
from dataset_u import *
from train_u import train
from predict_u import predict
from model import DAGTransformer
from torch.utils.data import DataLoader
import time
import random
import numpy as np
import torch
import wandb
from argparse import ArgumentParser
import json


class Ubackprop:
    def __init__(self, dag, config, data, model_file, new_model_file, mask, n_uniform_variables, seed=None):
        self.model = DAGTransformer(dag=dag, **config['model'])
        self.data = pd.read_csv(data)
        self.dag = dag
        self.train_config = config['training']
        self.model_file = model_file
        self.new_model_file = new_model_file
        self.mask = mask
        self.n_uniform_variables = n_uniform_variables
        self.seed = seed if seed is not None else random.randint(0, 10000)

        self.dataset = CausalDataset(self.data, self.dag, add_u=True, n_uniform_variables=self.n_uniform_variables,
                                seed_value=self.seed)
        # print distribution of U1
        #print(self.data['U1'].describe())

        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.train_config['batch_size'],
                                     shuffle=True, collate_fn=self.dataset.collate_fn)

        self.initial_train()

    def initial_train(self):
        train(model=self.model,
              data=self.data,
              dataloader=self.dataloader,
              config=self.train_config,
              mask=self.mask,
              model_file=self.model_file)
        print("Initial training complete.")

    def back_propagate(self, lr=0.0001, n_iters=1000, reg_lambda=0.01, grad_clip=1.0):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.load_state_dict(torch.load(self.model_file))

        wandb.init(project="DAG transformer u backpropagation", config=self.train_config,
                   tags=[self.train_config['project_tag']])

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Set requires_grad to False for all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Set requires_grad to True only for parameters whose names start with 'U'
        for name, param in self.model.named_parameters():
            if 'U' in name:
                param.requires_grad = True

        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)

        for i in range(n_iters):
            for batch in self.dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(batch, mask=True)
                batch_loss, batch_items = causal_loss_fun(outputs, batch, return_items=True)

                # Add L2 regularization to the loss
                reg_loss = 0
                for name, param in self.model.named_parameters():
                    if 'U' in name:
                        reg_loss += torch.norm(param, p=2)
                batch_loss += reg_lambda * reg_loss

                batch_loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                opt.step()
                opt.zero_grad()

                for item in batch_items.keys():
                    wandb.log({item: batch_items[item]})

                wandb.log({'loss': batch_loss.item()})

            scheduler.step()

        # Create a dictionary to store the final 'U' parameters
        final_u_params = {}

        # Store the final 'U' parameters in the dictionary
        for name, param in self.model.named_parameters():
            if 'U' in name:
                final_u_params[name] = param.detach().cpu().numpy()

        # Update the U parameters in the model with the final U parameters
        for name, param in self.model.named_parameters():
            if 'U' in name:
                param.data = torch.from_numpy(final_u_params[name]).to(param.device)

        # Save the updated model
        torch.save(self.model.state_dict(), self.new_model_file)

        # Return the final 'U' parameters
        return final_u_params

    def get_new_model(self):
        new_model = self.model.load_state_dict(torch.load(self.new_model_file))
        return new_model



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dag', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--model_file', type=str, required=True)
    parser.add_argument('--new_model_file', type=str, required=True)
    parser.add_argument('--mask', type=bool, required=True)
    parser.add_argument('--n_uniform_variables', type=int, required=True)
    parser.add_argument('--seed', type=int, required=False)

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

    model = Ubackprop(dag=dag,
                          config=config,
                          data=args.data_file,
                          model_file=args.model_file,
                          new_model_file=args.new_model_file,
                          mask=args.mask,
                          n_uniform_variables=args.n_uniform_variables,
                          seed=args.seed)



    # Run the back-propagation to obtain the final U parameters
    final_u_params = model.back_propagate(lr=0.0001, n_iters=100, reg_lambda=0.01, grad_clip=1.0)

    new_model = model.get_new_model()

    # Extract the U values from the final parameters
    u_values = {}
    for name, param in final_u_params.items():
        if 'U' in name:
            u_values[name] = param

    # Print the U values
    for name, value in u_values.items():
        # print shape of U variable
        print(f"U variable: {name}, shape: {value.shape}")
        print(f"U variable: {name}, value: {value}")




