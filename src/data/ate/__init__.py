from typing import Dict, Any, Optional, Tuple, Union

from sklearn.preprocessing import StandardScaler

from src.data.ate.preprocess import get_preprocessor_ate
from src.data.ate.demand_pv import generate_test_demand_pv, generate_train_demand_pv
from src.data.ate.dsprite import generate_train_dsprite, generate_test_dsprite
from src.data.ate.data_class import PVTestDataSet, PVTrainDataSet, RHCTestDataSet
from src.data.ate.rhc_experiment import generate_train_rhc, generate_val_rhc, generate_test_rhc


def generate_train_data_ate(data_config: Dict[str, Any], rand_seed: int) -> PVTrainDataSet:
    data_name = data_config["name"]
    if data_name == "demand":
        return generate_train_demand_pv(seed=rand_seed, **data_config)
    elif data_name == "dsprite":
        return generate_train_dsprite(rand_seed=rand_seed, **data_config)
    elif data_name == 'rhc':
        return generate_train_rhc(data_config['use_all_X'].lower() == "true")  # no random seed needed for this one
    else:
        raise ValueError(f"data name {data_name} is not valid")


def generate_val_data_ate(data_config: Dict[str, Any], rand_seed: int) -> PVTrainDataSet:
    data_name = data_config["name"]
    if data_name == "dsprite":
        n_sample = data_config["val_sample"]
        return generate_train_dsprite(rand_seed=rand_seed, n_sample=n_sample)
    elif data_name == "demand":
        return generate_train_demand_pv(seed=rand_seed, **data_config)
    elif data_name == "rhc":
        return generate_val_rhc(data_config['use_all_X'].lower() == "true")
    else:
        raise ValueError(f"data name {data_name} is not valid")


def generate_test_data_ate(data_config: Dict[str, Any]) -> Optional[Union[PVTestDataSet, RHCTestDataSet]]:
    data_name = data_config["name"]
    if data_name == "demand":
        return generate_test_demand_pv(**data_config)
    elif data_name == "dsprite":
        return generate_test_dsprite()
    elif data_name == "rhc":
        return generate_test_rhc(data_config['use_all_X'].lower() == "true")
    else:
        raise ValueError(f"data name {data_name} is not valid")


def standardise(data: PVTrainDataSet) -> Tuple[PVTrainDataSet, Dict[str, StandardScaler]]:
    treatment_proxy_scaler = StandardScaler()
    treatment_proxy_s = treatment_proxy_scaler.fit_transform(data.treatment_proxy)

    treatment_scaler = StandardScaler()
    treatment_s = treatment_scaler.fit_transform(data.treatment)

    outcome_scaler = StandardScaler()
    outcome_s = outcome_scaler.fit_transform(data.outcome)

    outcome_proxy_scaler = StandardScaler()
    outcome_proxy_s = outcome_proxy_scaler.fit_transform(data.outcome_proxy)

    backdoor_s = None
    backdoor_scaler = None
    if data.backdoor is not None:
        backdoor_scaler = StandardScaler()
        backdoor_s = backdoor_scaler.fit_transform(data.backdoor)

    train_data = PVTrainDataSet(treatment=treatment_s,
                                treatment_proxy=treatment_proxy_s,
                                outcome_proxy=outcome_proxy_s,
                                outcome=outcome_s,
                                backdoor=backdoor_s)

    scalers = dict(treatment_proxy_scaler=treatment_proxy_scaler,
                   treatment_scaler=treatment_scaler,
                   outcome_proxy_scaler=outcome_proxy_scaler,
                   outcome_scaler=outcome_scaler,
                   backdoor_scaler=backdoor_scaler)

    return train_data, scalers
