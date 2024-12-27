# DAG-aware Transformer for Causal Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)

## Overview

This repository contains the implementation of a DAG-aware Transformer model for causal inference, as described in our paper [insert paper title and link when available]. Our model incorporates causal structure into the attention mechanism, allowing for more accurate modeling of causal relationships in various estimation frameworks including G-formula, Inverse Probability Weighting (IPW), and Augmented Inverse Probability Weighting (AIPW).

## Key Features

- DAG-aware attention mechanism
- Support for multiple causal inference methods (G-formula, IPW, AIPW)
- Flexible architecture for both joint and separate training of propensity score and outcome models
- Extension to proximal causal inference

## Project Structure

Our project is organized as follows:
```
.
├── README.md
├── config
│   ├── dag
│   └── train
├── data
│   ├── acic
│   └── lalonde
├── experiments
│   ├── results
│   └── tuning
├── requirements.txt
├── scripts
│   ├── myjob.sh
│   └── myjob_proximal.sh
├── src
│   ├── data
│   ├── dataset.py
│   ├── evaluate
│   ├── experiment.py
│   ├── experiment_proximal.py
│   ├── models
│   ├── train
│   ├── utils.py
│   ├── utils_proximal
│   └── visualization
└── tests
```

- `config/`: Contains configuration files for DAG structures and training parameters.
- `data/`: Contains data loading and preprocessing scripts.
- `experiments/`: Holds experimental results.
- `scripts/`: Contains scripts for running the experiments.
- `src/`: The main source code directory.
  - `data/`: Data loading and preprocessing modules.
  - `evaluate/`: Evaluation metrics and functions.
  - `models/`: DAG-aware Transformer model and baseline models along with their loss functions.
  - `train/`: Programs to compute pseudo ATE/CATE (see descriptions in Hyper-parameter tuning section in our paper) 
  and the computed values. 
  - `utils/`: Utility functions for data processing and model training.
  - `utils_proximal/`: Utility functions for proximal inference.
  - `visualization/`: Code for generating plots and visualizations.
  - `experiment.py`: Main script for running experiments.
  - `experiment_proximal.py`: Main script for running proximal inference experiments.
- `tests/`: Unit tests for the project.


## Installation

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Datasets

We evaluate our model on four datasets:

1. Lalonde-CPS
2. Lalonde-PSID
3. ACIC
4. Demand dataset (for proximal inference)

Data preprocessing scripts and instructions can be found in the `data/` directory.

## Experiments

### Lalonde-CPS, Lalonde-PSID and ACIC

To reproduce the experiments for Lalonde-CPS, Lalonde-PSID and ACIC, run:

```bash
python3 src/experiment.py \
        --config config/train/<DATA_NAME>/<DATA_NAME>_sample<SAMPLE_ID>.json \
        --dag <DAG_TYPE> \
        --estimator <ESTIMATOR_TYPE> \
        --data_name <DATA_NAME>[myjob.sh](scripts%2Fmyjob.sh)
```

#### Parameters

- **CONFIG_FILE**: The configuration file for the experiment
  - Location: `config/train/<DATA_NAME>/`
  - Naming Convention: `<DATA_NAME>_sample<SAMPLE_ID>.json`
  - Examples:
    - `acic_sample1.json`
    - `lalonde_cps_sample2.json`
    - `lalonde_psid_sample3.json`

- **DAG_TYPE**: The type of Directed Acyclic Graph (DAG) to use
  - Options:
    - `dag_g_formula`
    - `dag_ipw`
    - `dag_aipw`

- **ESTIMATOR_TYPE**: The type of estimator to use
  - Options:
    - `g-formula`
    - `ipw`
    - `aipw`

- **DATA_NAME**: The name of the dataset
  - Options:
    - `lalonde_cps`
    - `lalonde_psid`
    - `acic`

- **SAMPLE_ID**: The sample ID for the experiment
  - A numeric value from 1 to 10 (e.g., 1, 2, 3, ...)

#### Example

```bash
python3 src/experiment.py \
        --config config/train/lalonde_cps/lalonde_cps_sample1.json \
        --dag dag_ipw \
        --estimator ipw \
        --data_name lalonde-cps
```


### Demand Dataset (Proximal Inference)

```bash
python3 src/experiment_proximal.py \
        --dag <DAG_CONFIG_FILE> \
        --config config/train/proximal/nmmr_<STATISTICS>_z_transformer_n<SAMPLE_SIZE>.json \
        --results_dir <RESULTS_DIRECTORY> \
        --sample_index <SAMPLE_INDEX>
```

#### Parameters

- **DAG_CONFIG_FILE**: The configuration file for the Directed Acyclic Graph (DAG)
  - Location: `config/dag/`
  - Example: `proximal_dag_z.json`

- **STATISTICS**: The type of statistics used in proximal inference
  - Options: `u` (U-statistics) or `v` (V-statistics)

- **SAMPLE_SIZE**: The size of the sample used in the experiment
  - Example values: `50000`, `100000`, etc.

- **RESULTS_DIRECTORY**: The directory where results will be stored
  - Default: `experiments/results/proximal`

- **SAMPLE_INDEX**: The index of the sample to use for the experiment (form 0 to 19)
  - Example values: `0`, `1`, `2`, etc.

#### Example

```bash
python3 src/experiment_proximal.py \
        --dag config/dag/proximal_dag_z.json \
        --config config/train/proximal/nmmr_v_z_transformer_n50000.json \
        --results_dir experiments/results/proximal \
        --sample_index 1
```
You can also run the experiment using the provided script `scripts/myjob.sh` for lalonde-cps, lalonde-acic and ACIC; and
`scripts/myjob_proximal.sh` for demand by modifying the parameters in the script.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{liu2024dagawaretransformercausaleffect,
      title={DAG-aware Transformer for Causal Effect Estimation}, 
      author={Manqing Liu and David R. Bellamy and Andrew L. Beam},
      year={2024},
      eprint={2410.10044},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2410.10044}, 
}
```

## License

This project is licensed under the MIT License. For the complete terms and conditions, refer to the [LICENSE](LICENSE) file or visit:
[https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT).


## Contact

For any questions or concerns, please open an issue or contact Manqing Liu at manqingliu@g.harvard.edu.