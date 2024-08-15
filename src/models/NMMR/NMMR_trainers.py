import os.path as op
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np
from src.dataset import *

import torch
from torch.utils.data import DataLoader as Dataloader
import torch.optim as optim
import torch.nn as nn
import tqdm
import wandb

from src.data.ate.data_class import PVTrainDataSetTorch, PVTestDataSetTorch
from src.utils import predict_function
from src.models.NMMR.NMMR_loss import NMMR_loss, NMMR_loss_batched, NMMR_loss_transformer
from src.models.NMMR.NMMR_model import MLP_for_NMMR
from src.models.NMMR.kernel_utils import calculate_kernel_matrix, calculate_kernel_matrix_batched, rbf_kernel
from src.models.transformer_model import DAGTransformer


class NMMR_Trainer_DemandExperiment(object):
    def __init__(self,
                 configs: Dict[str, Any],
                 data_configs: Dict[str, Any],
                 dag: Dict[str, Any],
                 train_config_transformer: Dict[str, Any],
                 model_config_transformer: Dict[str, Any],
                 train_config_mlp: Dict[str, Any],
                 model_config_mlp: Dict[str, Any],
                 random_seed: int,
                 dump_folder: Optional[Path] = None):

        wandb.init(project="DAG transformer",
                   config=configs,
                   tags='proximal_demand')

        self.configs = configs
        self.data_config = data_configs
        self.dag = dag
        self.train_config_transformer = train_config_transformer
        self.model_config_transformer = model_config_transformer
        self.train_config_mlp = train_config_mlp
        self.model_config_mlp = model_config_mlp
        self.n_sample = self.data_config['n_sample']
        self.gpu_flg = torch.cuda.is_available()

        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def compute_kernel(self, kernel_inputs):
        return calculate_kernel_matrix(kernel_inputs)

    def train_transformer(self,
          train_dataloader: Dataloader,
          val_data: CausalDataset,
          val_dataloader: Dataloader,
          test_dataloader: Dataloader) -> DAGTransformer:

        model = DAGTransformer(dag=self.dag,
                            **self.model_config_transformer)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f'Using device: {device}')

        optimizer = optim.Adam(list(model.parameters()), lr=self.train_config_transformer['learning_rate'],
                               weight_decay=self.train_config_transformer['l2_penalty'])

        epoch_bar = tqdm.tqdm(range(self.train_config_transformer['n_epochs']), desc="Training", position=0, leave=True)
        bin_left_edges = {k: torch.tensor(v[:-1], dtype=torch.float32).to(device) for k, v in train_dataloader.dataset.bin_edges.items()}
        for epoch in epoch_bar:
            model.train(True)
            batch_bar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}", position=1, leave=False)
            for batch_raw, batch_binned in batch_bar: 
                optimizer.zero_grad()
                batch_binned = {k: v.to(device) for k, v in batch_binned.items()}
                # send batch_raw to device
                batch_raw = {k: v.to(device) for k, v in batch_raw.items()}
                
                # TODO: check masking during forward pass
                outputs = model(batch_binned, mask="True")

                # Transform the model outputs back to the original scale
                transformed_outputs = {}
                for output_name, output in outputs.items():
                    # softmax_output = torch.softmax(output, dim=1)  # TODO: remove/replace after testing
                    weighted_avg = torch.sum(output * bin_left_edges[output_name].to(device), dim=1, keepdim=True)
                    transformed_outputs[output_name] = weighted_avg

                labels = {'outcome': batch_raw['outcome']}
                treatment = {'treatment': batch_raw['treatment']}
                treatment_binned = {'treatment': batch_binned['treatment']}
                kernel_inputs_train = torch.cat((batch_raw['treatment'], batch_raw['treatment_proxy1'],
                                                batch_raw['treatment_proxy2']), dim=1)
                kernel_matrix_train = self.compute_kernel(kernel_inputs_train)

                model_output = transformed_outputs['outcome']
                model_output = model_output.to(device)
                target = labels['outcome']
                target = target.to(device)
                alpha = self.train_config_transformer['alpha']

                batch_loss, batch_items = NMMR_loss_transformer(model_output,
                                                                target,
                                                                treatment_binned,
                                                                kernel_matrix_train,
                                                                loss_name=self.train_config_transformer['loss_name'],
                                                                alpha=0.01,
                                                                return_items=True)
                batch_loss.backward()
                optimizer.step()
                for item in batch_items.keys():
                    wandb.log({f"train_{item}": batch_items[item]})

                batch_bar.set_postfix(loss=batch_loss.item())
            
            batch_bar.close()

            model.eval()
            with torch.no_grad():
                for batch_raw, batch_binned in val_dataloader:
                    batch_binned = {k: v.to(device) for k, v in batch_binned.items()}
                    batch_raw = {k: v.to(device) for k, v in batch_raw.items()}
                    outputs = model(batch_binned, mask="True")

                    # Transform the model outputs back to the original scale
                    transformed_outputs = {}
                    for output_name, output in outputs.items():
                        # softmax_output = torch.softmax(output, dim=1)
                        weighted_avg = torch.sum(output * bin_left_edges[output_name].to(device), dim=1, keepdim=True)
                        transformed_outputs[output_name] = weighted_avg

                    labels = {'outcome': batch_raw['outcome']}
                    treatment_binned = {'treatment': batch_binned['treatment']}
                    kernel_inputs_val = torch.cat((batch_raw['treatment'], batch_raw['treatment_proxy1'],
                                                    batch_raw['treatment_proxy2']), dim=1)
                    kernel_matrix_val = self.compute_kernel(kernel_inputs_val)

                    model_output = transformed_outputs['outcome']
                    model_output = model_output.to(device)
                    target = labels['outcome']
                    target = target.to(device)

                    batch_loss, batch_items = NMMR_loss_transformer(model_output,
                                                                    target,
                                                                    treatment_binned,
                                                                    kernel_matrix_val,
                                                                    loss_name=self.train_config_transformer['loss_name'],
                                                                    alpha=0.01,
                                                                    return_items=True)
                    for item in batch_items.keys():
                        wandb.log({f"val_{item}": batch_items[item]})

                E_w_haw, oos_loss, E_ydoA, predictions = self.predict_transformer(model, self.data_config, val_data, self.dag, self.n_sample, test_dataloader)
                wandb.log({'OOS loss transformer': oos_loss, 'E_w_haw_transformer': E_w_haw, 'E_ydoA': E_ydoA, 'predictions': predictions})

            epoch_bar.set_postfix(val_loss=batch_loss.item())
        
        epoch_bar.close()

        return model

    @staticmethod
    def predict_transformer(model: DAGTransformer,
                data_config: Dict[str, Any],
                val_data: CausalDataset,
                dag: Dict[str, Any],
                n_sample: int,
                test_dataloader: Dataloader):
        # Create a 3-dim array with shape [intervention_array_len, n_samples, 2]
        # This will contain the test values for do(A) chosen by Xu et al. as well as {n_samples} random draws for W
        # Compute model's predicted E[Y | do(A)] = E_w[h(a, w)]
        # Note: the mean is taken over the n_sample axis, so we obtain {intervention_array_len} number of expected values

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        bin_left_edges = {k: np.array(v[:-1], dtype=np.float32) for k, v in test_dataloader.dataset.bin_edges.items()}
        softmaxes = []
        with torch.no_grad():
            for _, batch_binned in test_dataloader:
                batch_binned = {k: v.to(device) for k, v in batch_binned.items()}
                outputs = model(batch_binned, mask="True")    
                output = outputs['outcome'].cpu().numpy()
                softmaxes.append(output)

        # concatenate and unbin the predictions
        # TODO: verify the tensor operations here for computing oos_loss
        softmaxes = np.concatenate(softmaxes, axis=0)
        predictions = np.sum(softmaxes * bin_left_edges['outcome'], axis=1)
        predictions_reshape = predictions.reshape(n_sample, 10, order='F')
        E_w_haw = np.mean(predictions_reshape, axis=0)
        test_data, E_ydoA = make_test_data(data_config, val_data, dag)
        oos_loss = np.mean((E_w_haw - E_ydoA) ** 2)

        return E_w_haw, oos_loss, E_ydoA, predictions


    def train_mlp(self,
                  train_t: PVTrainDataSetTorch,
                  test_data_t: PVTestDataSetTorch,
                  val_data_t: PVTrainDataSetTorch,
                  verbose: int = 0) -> MLP_for_NMMR:

        # inputs consist of (A, W) tuples
        model = MLP_for_NMMR(input_dim=2, train_params=self.model_config_mlp)

        if self.gpu_flg:
            train_t = train_t.to_gpu()
            model.cuda()

        # weight_decay implements L2 penalty
        optimizer = optim.Adam(list(model.parameters()), lr=self.train_config_mlp['learning_rate'],
                               weight_decay=self.train_config_mlp['l2_penalty'])

        # train model
        for epoch in range(self.train_config_mlp['n_epochs']):
            permutation = torch.randperm(self.n_sample)

            for i in range(0, self.n_sample, self.train_config_mlp['batch_size']):
                indices = permutation[i:i + self.train_config_mlp['batch_size']]
                batch_A, batch_W, batch_y = train_t.treatment[indices], train_t.outcome_proxy[indices], \
                                            train_t.outcome[indices]

                optimizer.zero_grad()
                batch_x = torch.cat((batch_A, batch_W), dim=1)
                pred_y = model(batch_x)
                kernel_inputs_train = torch.cat((train_t.treatment[indices], train_t.treatment_proxy[indices]), dim=1)
                kernel_matrix_train = self.compute_kernel(kernel_inputs_train)

                causal_loss_train = NMMR_loss(pred_y, batch_y, kernel_matrix_train, self.train_config_mlp['loss_name'])
                causal_loss_train.backward()
                optimizer.step()
                wandb.log({'Training loss mlp': causal_loss_train})

            model.eval()
            with torch.no_grad():

                preds_val = model(torch.cat((val_data_t.treatment, val_data_t.outcome_proxy), dim=1))

                # compute the full kernel matrix
                kernel_inputs_val = torch.cat((val_data_t.treatment, val_data_t.treatment_proxy), dim=1)
                kernel_matrix_val = self.compute_kernel(kernel_inputs_val)

                # calculate and log the causal loss (validation)
                causal_loss_val = NMMR_loss(preds_val, val_data_t.outcome, kernel_matrix_val, self.train_config_mlp['loss_name'])

                wandb.log({'Validation loss mlp': causal_loss_val})

                E_w_haw, oos_loss, predictions = self.predict_mlp(model, self.data_config, test_data_t, val_data_t)
                # print E_w_haw for each epoch
                print(f"Epoch: {epoch}, E_w_haw_mlp: {E_w_haw}")
                # print oos loss for each epoch
                print(f"Epoch: {epoch}, OOS loss mlp: {oos_loss.item()}")
                wandb.log({'OOS loss mlp': oos_loss, "MLP predictions": predictions})

        return model

    @staticmethod
    def predict_mlp(model: MLP_for_NMMR,
                    data_config: Dict[str, Any],
                    test_data_t: PVTestDataSetTorch,
                    val_data_t: PVTrainDataSetTorch):
        # Create a 3-dim array with shape [intervention_array_len, n_samples, 2]
        # This will contain the test values for do(A) chosen by Xu et al. as well as {n_samples} random draws for W
        test_data = generate_test_data_ate(data_config=data_config)
        intervention_array_len = test_data_t.treatment.shape[0]
        num_W_test = val_data_t.outcome_proxy.shape[0]
        temp1 = test_data_t.treatment.expand(-1, num_W_test)
        temp2 = val_data_t.outcome_proxy.expand(-1, intervention_array_len)
        model_inputs_test = torch.stack((temp1, temp2.T), dim=-1)

        # Compute model's predicted E[Y | do(A)] = E_w[h(a, w)]
        # Note: the mean is taken over the n_sample axis, so we obtain {intervention_array_len} number of expected values
        with torch.no_grad():
            mlp_predictions = model(model_inputs_test)
            E_w_haw = torch.mean(mlp_predictions, dim=1)
            oos_loss = torch.mean((E_w_haw - test_data.structural) ** 2)

        return E_w_haw.cpu(), oos_loss, mlp_predictions.cpu()
