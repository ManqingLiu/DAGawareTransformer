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

        #if self.gpu_flg:
        #    train_dataloader = train_dataloader.to_gpu()
        #    val_dataloader = val_dataloader.to_gpu()
        #model.cuda()

        # weight_decay implements L2 penalty
        optimizer = optim.Adam(list(model.parameters()), lr=self.learning_rate, weight_decay=self.l2_penalty)

        # train model
        for epoch in tqdm(range(self.n_epochs)):
            for batch in train_dataloader:
                optimizer.zero_grad()
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch, mask=self.mask)
                labels = {'outcome': batch['outcome']}
                # To Do: should return values to original scales before passing to compute_kernel (right now they are in bins)
                #print(batch['treatment'].shape)  # torch.Size([1000, 1])
                kernel_inputs_train =torch.cat((batch['treatment'], batch['treatment_proxy1'],
                                                batch['treatment_proxy2']), dim=1)
                kernel_matrix_train = self.compute_kernel(kernel_inputs_train)
                #print(kernel_matrix_train.shape)  # torch.Size([1000, 1000])
                # To Do: should unbin the model output to original scale before passing it to NMMR_loss_transformer
                model_output = outputs['outcome']  # of shape [1000, 10]
                print(model_output)
                #print(model_output.shape)
                target = labels['outcome']
                print(target)
                #print(target.shape)  # of shape [1000, 1]

                loss, batch_items = NMMR_loss_transformer(model_output, target, kernel_matrix_train,
                                                                loss_name=self.loss_name, return_items=True)
                loss.backward()
                optimizer.step()


        '''
        # at the end of each epoch, log metrics
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch, mask=self.mask)
                #print(outputs['outcome'].shape)  torch.Size([100, 10]) -> [batch_size, num_bins]
                labels = {'outcome': batch['outcome']}
                kernel_inputs_val = torch.cat((batch['treatment'], batch['treatment_proxy1'],
                                               batch['treatment_proxy2']), dim=1)
                kernel_matrix_val = self.compute_kernel(kernel_inputs_val)
                model_output = outputs['outcome']
                target = labels['outcome']
                val_loss, batch_items = NMMR_loss_transformer(model_output, target, kernel_matrix_val,
                                                loss_name=self.loss_name, return_items=True)
                print(f"Validation Loss: {val_loss}")
        '''

        return model

    @staticmethod
    def predict(model,
                n_sample,
                model_intput_test_data,
                model_input_test_dataloader):
        # Create a 3-dim array with shape [intervention_array_len, n_samples, 2]
        # This will contain the test values for do(A) chosen by Xu et al. as well as {n_samples} random draws for W
        # Compute model's predicted E[Y | do(A)] = E_w[h(a, w)]
        # Note: the mean is taken over the n_sample axis, so we obtain {intervention_array_len} number of expected values
        #model = DAGTransformer(dag=self.dag,
        #                       **self.model_config)


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        model = model.to(device)


        predictions = []
        with torch.no_grad():
            for batch in model_input_test_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch, mask=True)
                batch_predictions = []
                for output_name in outputs.keys():
                    # Detach the outputs and move them to cpu
                    output = outputs[output_name].cpu().numpy()
                    output = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
                    # Append the reshaped output to batch_predictions
                    batch_predictions.append(output)
                # concatenate the batch predictions along the second axis
                batch_predictions = np.concatenate(batch_predictions, axis=1)
                predictions.append(batch_predictions)

        # assign column names to the predictions_df
        predictions = np.concatenate(predictions, axis=0)
        prediction_transformer = PredictionTransformer(model_intput_test_data.bin_edges)
        transformed_predictions = prediction_transformer.transform_proximal(predictions, n_sample)
        #print(transformed_predictions.shape)  #[10, 1000, 1]

        E_w_haw = torch.mean(transformed_predictions, dim=1)  # should return 10 expected values

        return E_w_haw.cpu()


