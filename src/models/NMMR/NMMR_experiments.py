import os.path as op
from pathlib import Path
from typing import Dict, Any
from argparse import ArgumentParser
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from src.dataset import CausalDataset
from src.utils_proximal import filter_dataset

from src.data.ate import generate_train_data_ate, generate_val_data_ate, generate_test_data_ate, get_preprocessor_ate
from src.data.ate.data_class import PVTrainDataSet, PVTrainDataSetTorch, PVTestDataSetTorch, RHCTestDataSetTorch
from src.models.NMMR.NMMR_trainers import NMMR_Trainer_DemandExperiment

def NMMR_experiment(data_config: Dict[str, Any],
                    model_config: Dict[str, Any],
                    train_config: Dict[str, Any],
                    dag: Dict[str, Any],
                    mask: bool,
                    one_mdl_dump_dir: Path,
                    random_seed: int = 42,
                    verbose: int = 0):
    # set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # generate train data
    train_data = generate_train_data_ate(data_config=data_config, rand_seed=random_seed)
    val_data = generate_val_data_ate(data_config=data_config, rand_seed=random_seed + 1)
    test_data = generate_test_data_ate(data_config=data_config)


    # convert datasets to Torch (for GPU runtime)
    train_t = PVTrainDataSetTorch.from_numpy(train_data)
    val_data_t = PVTrainDataSetTorch.from_numpy(val_data)

    data_name = data_config.get("name", None)
    if data_name in ['dsprite', 'demand']:
        test_data_t = PVTestDataSetTorch.from_numpy(test_data)
    elif data_name == 'rhc':
        test_data_t = RHCTestDataSetTorch.from_numpy(test_data)
    else:
        raise KeyError(f"Your data config contained name = {data_name}, but must be one of [dsprite, demand, rhc]")

    intervention_array_len = test_data_t.treatment.shape[0]
    num_W_test = val_data_t.outcome_proxy.shape[0]
    treatment = test_data_t.treatment.expand(-1, num_W_test)
    treatment_proxy1 = val_data_t.treatment_proxy[:, 0].unsqueeze(1).expand(-1, intervention_array_len)
    treatment_proxy2 = val_data_t.treatment_proxy[:, 1].unsqueeze(1).expand(-1, intervention_array_len)
    outcome_proxy = val_data_t.outcome_proxy.expand(-1, intervention_array_len)
    outcome = val_data_t.outcome.expand(-1, intervention_array_len)
    model_inputs_test = torch.stack((treatment,
                                     treatment_proxy1.T,
                                     treatment_proxy2.T,
                                     outcome_proxy.T,
                                     outcome.T), dim=-1)


    train_data_dict = {
        'treatment': train_t.treatment,
        'treatment_proxy1': train_t.treatment_proxy[:, 0],
        'treatment_proxy2': train_t.treatment_proxy[:, 1],
        'outcome_proxy': train_t.outcome_proxy,
        'outcome': train_t.outcome
    }

    val_data_dict = {
        'treatment': val_data.treatment,
        'treatment_proxy1': val_data.treatment_proxy[:, 0],
        'treatment_proxy2': val_data.treatment_proxy[:, 1],
        'outcome_proxy': val_data.outcome_proxy,
        'outcome': val_data.outcome
    }

    model_input_test_dict = {
        'treatment': model_inputs_test[:, :, 0],
        'treatment_proxy1': model_inputs_test[:, : ,1],
        'treatment_proxy2': model_inputs_test[:, : ,2],
        'outcome_proxy': model_inputs_test[:, :, 3],
        'outcome': model_inputs_test[:, : ,4]
    }



    train_data = CausalDataset(train_data_dict, dag)
    val_data = CausalDataset(val_data_dict, dag)
    model_input_test_data = CausalDataset(model_input_test_dict, dag)


    train_dataloader = DataLoader(train_data,
                            batch_size=train_config['batch_size'],
                            shuffle=True,
                            collate_fn=train_data.collate_fn)

    val_dataloader = DataLoader(val_data,
                            batch_size=train_config['batch_size'],
                            shuffle=True,
                            collate_fn=val_data.collate_fn)

    model_input_test_dataloader = DataLoader(model_input_test_data,
                            batch_size=train_config['batch_size'],
                            shuffle=True,
                            collate_fn=model_input_test_data.collate_fn)

    # retrieve the trainer for this experiment
    if data_name == "demand":
        trainer = NMMR_Trainer_DemandExperiment(data_config,
                                                dag,
                                                train_config,
                                                model_config,
                                                mask,
                                                random_seed,
                                                one_mdl_dump_dir)

    # train model
    model = trainer.train(train_dataloader, val_dataloader, verbose)

    # prepare test data on the gpu
    #if trainer.gpu_flg:
    #    torch.cuda.empty_cache()
    #    model_input_test_data = model_input_test_data.to_gpu()

    n_sample = data_config.get("n_sample", None)

    if data_name == "demand":
        E_wx_hawx = trainer.predict(model, n_sample,
                                    model_input_test_data,
                                    model_input_test_dataloader)


    pred = E_wx_hawx.detach().numpy()
    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)

    if hasattr(test_data, 'structural'):
        # test_data.structural is equivalent to EY_doA
        np.testing.assert_array_equal(pred.shape, test_data.structural.shape)
        oos_loss = np.mean((pred - test_data.structural) ** 2)
    else:
        oos_loss = None

    if trainer.log_metrics:
        return oos_loss, pd.DataFrame(
            data={'causal_loss_train': torch.Tensor(trainer.causal_train_losses[-50:], device="cpu").numpy(),
                  'causal_loss_val': torch.Tensor(trainer.causal_val_losses[-50:], device="cpu").numpy()})
    else:
        return oos_loss

if __name__ == '__main__':
    parser = ArgumentParser()
    # Load the configurations from the JSON file
    with open(Path('../../../config/train/proximal/nmmr_u_transformer_n1000.json'), 'r') as f:
        config = json.load(f)

    # Extract the data and model configurations
    data_config = config['data']
    train_config = config['training']
    model_config = config['model']

    # Define the directory where the model will be saved
    one_mdl_dump_dir = Path("../../../experiments/results/proximal")

    args = parser.parse_args()

    with open(Path('../../../config/dag/proximal_dag.json'), 'r') as f:
        #print(f'Loading dag file from {args.dag}')
        dag = json.load(f)

    num_nodes = len(dag['nodes'])
    dag['node_ids'] = dict(zip(dag['nodes'], range(num_nodes)))
    mask= True

    # Run the experiment
    oos_loss = NMMR_experiment(data_config, model_config, train_config, dag, mask, one_mdl_dump_dir)
    # Print the oos_loss
    print(f"Out of sample loss: {oos_loss}")

