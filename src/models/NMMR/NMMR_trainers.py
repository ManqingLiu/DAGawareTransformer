import os.path as op
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np
from src.dataset import *

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader as Dataloader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from src.data.ate.data_class import PVTrainDataSetTorch, PVTestDataSetTorch, RHCTestDataSet
from src.models.NMMR.NMMR_loss import NMMR_loss, NMMR_loss_batched, NMMR_loss_transformer
from src.models.NMMR.NMMR_model import MLP_for_NMMR, cnn_for_dsprite
from src.models.NMMR.kernel_utils import calculate_kernel_matrix, calculate_kernel_matrix_batched, rbf_kernel
from src.models.NMMR.transformer_model import DAGTransformer, causal_loss_fun


class NMMR_Trainer_DemandExperiment(object):
    def __init__(self,
                 data_configs: Dict[str, Any],
                 dag: Dict[str, Any],
                 train_config: Dict[str, Any],
                 model_config: Dict[str, Any],
                 mask: bool,
                 random_seed: int,
                 dump_folder: Optional[Path] = None):


        self.data_config = data_configs
        self.dag = dag
        self.mask = mask
        self.train_config = train_config
        self.model_config = model_config
        self.n_sample = self.data_config['n_sample']
        self.n_epochs = train_config['n_epochs']
        self.batch_size = train_config['batch_size']
        self.gpu_flg = torch.cuda.is_available()
        self.log_metrics = train_config['log_metrics'] == "True"
        self.l2_penalty = train_config['l2_penalty']
        self.learning_rate = train_config['learning_rate']
        self.loss_name = train_config['loss_name']
        self.num_layers = model_config['num_layers']
        self.dropout_rate = model_config['dropout_rate']
        self.embedding_dim = model_config['embedding_dim']
        self.num_heads = model_config['num_heads']
        self.name = model_config['name']

        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        if self.log_metrics:
            self.writer = SummaryWriter(log_dir=op.join(dump_folder, f"tensorboard_log_{random_seed}"))
            self.causal_train_losses = []
            self.causal_val_losses = []

    def compute_kernel(self, kernel_inputs):

        return calculate_kernel_matrix(kernel_inputs)

    def train(self,
          train_dataloader: Dataloader,
          val_dataloader: Dataloader,
          verbose: int = 0) -> DAGTransformer:

        model = DAGTransformer(dag=self.dag,
                            **self.model_config)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')

        optimizer = optim.Adam(list(model.parameters()), lr=self.learning_rate, weight_decay=self.l2_penalty)

        bin_left_edges = {k: torch.tensor(v[:-1], dtype=torch.float32).to(device) for k, v in train_dataloader.dataset.bin_edges.items()}

        for _ in tqdm(range(self.n_epochs)):
            for batch_raw, batch_binned in train_dataloader:
                optimizer.zero_grad()
                batch_binned = {k: v.to(device) for k, v in batch_binned.items()}
                outputs = model(batch_binned, mask=self.mask)

                # Transform the model outputs back to the original scale
                transformed_outputs = {}
                for output_name, output in outputs.items():
                    softmax_output = torch.softmax(output, dim=1)
                    weighted_avg = torch.sum(softmax_output * bin_left_edges[output_name].to(device), dim=1, keepdim=True)
                    transformed_outputs[output_name] = weighted_avg

                labels = {'outcome': batch_raw['outcome']}
                kernel_inputs_train = torch.cat((batch_raw['treatment'], batch_raw['treatment_proxy1'],
                                                batch_raw['treatment_proxy2']), dim=1)
                kernel_matrix_train = self.compute_kernel(kernel_inputs_train)

                model_output = transformed_outputs['outcome']
                target = labels['outcome']

                loss, _ = NMMR_loss_transformer(model_output, target, kernel_matrix_train,
                                                        loss_name=self.loss_name, return_items=True)
                print(f"Loss: {loss.item()}")
                loss.backward()
                optimizer.step()

        return model

    @staticmethod
    def predict(model,
                n_sample,
                bin_edges,
                test_dataloader):
        # Create a 3-dim array with shape [intervention_array_len, n_samples, 2]
        # This will contain the test values for do(A) chosen by Xu et al. as well as {n_samples} random draws for W
        # Compute model's predicted E[Y | do(A)] = E_w[h(a, w)]
        # Note: the mean is taken over the n_sample axis, so we obtain {intervention_array_len} number of expected values
        #model = DAGTransformer(dag=self.dag,
        #                       **self.model_config)


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        model = model.to(device)

        bin_left_edges = {k: np.array(v[:-1], dtype=np.float32) for k, v in test_dataloader.dataset.bin_edges.items()}
        softmaxes = []
        with torch.no_grad():
            for _, batch_binned in test_dataloader:
                batch_binned = {k: v.to(device) for k, v in batch_binned.items()}
                outputs = model(batch_binned, mask=True)
                batch_predictions = []
                for output_name in outputs.keys():
                    # Detach the outputs and move them to cpu
                    output = outputs[output_name].cpu().numpy()
                    output = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
                    # Append the reshaped output to batch_predictions
                    batch_predictions.append(output)
                # concatenate the batch predictions along the second axis
                batch_predictions = np.concatenate(batch_predictions, axis=1)
                softmaxes.append(batch_predictions)

        # assign column names to the predictions_df
        softmaxes = np.concatenate(softmaxes, axis=0)
        predictions = np.sum(softmaxes * bin_left_edges[output_name], axis=1)
        
        E_w_haw = np.mean(predictions.reshape(10, n_sample, 1), axis=1)  # should return 10 expected values

        return E_w_haw
