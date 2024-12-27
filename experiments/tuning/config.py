from ray import tune

config_test = {
    "filepaths": {
        "dag": "config/dag/acic_dag.json",
        "config": "config/train/acic/acic_sample3.json",
        "data_train_file": "data/acic/sample3/data_train_3.csv",
        "data_val_file": "data/acic/sample3/data_val_3.csv",
        "data_test_file": "data/acic/sample3/data_test_3.csv",
        "pseudo_cate_file": "src/train/acic/aipw_grf_pseudo_cate.csv",
        "result_file_ipw": "experiments/results/acic/acic_best_params_ipw_sample3.json",
        "result_file_cfcv": "experiments/results/acic/acic_best_params_cfcv_sample3.json"
    },
  "cfcv":{
        "training": {
            "num_epochs": 2,
            "batch_size": tune.choice([64, 128, 256, 512]),
            "learning_rate": tune.choice([1e-4, 1e-5, 1e-6]),
            "weight_decay": tune.choice([0, 1e-6, 1e-8, 1e-10]),
            "project_tag": "lalonde_cps",
            "dag_attention_mask": "True",
            "imbalance_loss_weight": tune.choice([100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0]),
            "prop_score_threshold": tune.choice([0, 0.1]),
            "eps": tune.choice([1e-6, 1e-8, 1e-10]),
            "time_limit": 14400
        },
        "model": {
            "num_layers": tune.choice([2, 4, 8, 16]),
            "dropout_rate": tune.choice([0, 0.1]),
            "embedding_dim": tune.choice([64, 128, 512]),
            "num_heads": tune.choice([4, 8, 16, 32, 64])
        }
    },
    "ipw":{
            "training": {
            "num_epochs": 100,
            "batch_size": tune.choice([256, 512, 1024]),
            "learning_rate": tune.choice([1e-9]),
            "weight_decay": tune.choice([0, 1e-6, 1e-8, 1e-10]),
            "project_tag": "lalonde_cps",
            "dag_attention_mask": "True",
            "imbalance_loss_weight": tune.choice([100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0]),
            "prop_score_threshold": tune.choice([0, 0.1]),
            "eps": tune.choice([1e-6, 1e-8, 1e-10]),
            "time_limit": 18000
        },
        "model": {
            "num_layers": tune.choice([2, 4, 8, 16]),
            "dropout_rate": tune.choice([0, 0.1]),
            "embedding_dim": tune.choice([64, 128, 512]),
            "num_heads": tune.choice([4, 8, 16, 32, 64])
        }
    },
  "random_seed": 42
}


config_psid = {
  "filepaths": {
    "dag": "config/dag/lalonde_psid_dag.json",
    "config": "config/train/lalonde_psid.json",
    "data_train_file": "data/lalonde/ldw_psid/lalonde_psid_train.csv",
    "data_val_file": "data/lalonde/ldw_psid/lalonde_psid_val.csv",
    "data_test_file": "data/lalonde/ldw_psid/lalonde_psid_test.csv",
    "output_file": "experiments/predict/lalonde_psid/predictions_psid.csv",
    "result_file_ipw": "config/train/lalonde_psid_best_params_ipw.json",
    "result_file_cfcv": "config/train/lalonde_psid_best_params_cfcv.json"
  },
  "cfcv":{
        "training": {
            "num_epochs": 100,
            "batch_size": tune.choice([32, 64, 128]),
            "learning_rate": tune.choice([1e-4, 1e-5]),
            "weight_decay": tune.choice([0, 1e-6, 1e-8, 1e-10]),
            "project_tag": "lalonde_psid",
            "dag_attention_mask": "True",
            "imbalance_loss_weight": tune.choice([100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01]),
            "prop_score_threshold": tune.choice([0, 0.1]),
            "eps": tune.choice([1e-6, 1e-8, 1e-10]),
            "time_limit": 18000
        },
        "model": {
            "num_layers": tune.choice([2, 4, 8, 16]),
            "dropout_rate": tune.choice([0, 0.1]),
            "embedding_dim": tune.choice([64, 128, 512]),
            "num_heads": tune.choice([4, 8, 16, 32, 64])
        }
    },
    "ipw":{
        "training": {
            "num_epochs": 100,
            "batch_size": tune.choice([32, 64, 128]),
            "learning_rate": tune.choice([1e-4, 1e-5]),
            "weight_decay": tune.choice([0, 1e-6, 1e-8, 1e-10]),
            "project_tag": "lalonde_psid",
            "dag_attention_mask": "True",
            "imbalance_loss_weight": tune.choice([100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01]),
            "prop_score_threshold": tune.choice([0, 0.1]),
            "eps": tune.choice([1e-6, 1e-8, 1e-10]),
            "time_limit": 18000
        },
        "model": {
            "num_layers": tune.choice([2, 4, 8, 16]),
            "dropout_rate": tune.choice([0, 0.1]),
            "embedding_dim": tune.choice([64, 128, 512]),
            "num_heads": tune.choice([4, 8, 16, 32, 64])
        }
    },
  "random_seed": 42
}


config_lalonde_cps = {
    "filepaths": {
        "dag": "config/dag/lalonde_cps_dag.json",
        "config": "config/train/lalonde_cps/lalonde_cps_sample0.json",
        "data_file": "data/lalonde/ldw_cps/ldw_cps.csv",
        "data_train_file": "data/lalonde/ldw_cps/sample0/train_data_0.csv",
        "data_val_file": "data/lalonde/ldw_cps/sample0/val_data_0.csv",
        "data_test_file": "data/lalonde/ldw_cps/sample0/test_data_0.csv",
        "pseudo_ate_file": "src/train/lalonde/lalonde_cps/aipw_grf_pseudo_ate.csv",
        "output_file": "experiments/predict/lalonde_cps/predictions_cps.csv",
        "result_file_ipw": "experiments/results/lalonde_cps/lalonde_cps_best_params_ipw_sample0.json",
        "result_file_cfcv": "experiments/results/lalonde_cps/lalonde_cps_best_params_cfcv_sample0.json"
  },
  "cfcv":{
        "training": {
            "num_epochs": tune.choice([20, 30, 40, 50]),
            "batch_size": tune.choice([64, 128, 256, 512]),
            "learning_rate": tune.choice([1e-4, 1e-5, 1e-6]),
            "weight_decay": tune.choice([0, 1e-3, 1e-6, 1e-8, 1e-10]),
            "project_tag": "lalonde_cps",
            "dag_attention_mask": "True",
            "imbalance_loss_weight": tune.choice([100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0]),
            "prop_score_threshold": tune.choice([0, 0.1]),
            "eps": tune.choice([1e-6, 1e-8, 1e-10]),
            "time_limit": 14400
        },
        "model": {
            "num_layers": tune.choice([2, 4, 8, 16]),
            "dropout_rate": tune.choice([0, 0.1, 0.2, 0.3]),
            "embedding_dim": tune.choice([64, 128, 512]),
            "num_heads": tune.choice([2, 4, 8, 16, 32, 64])
        }
    },
    "ipw":{
            "training": {
            "num_epochs": 100,
            "batch_size": tune.choice([256, 512, 1024]),
            "learning_rate": tune.choice([1e-9]),
            "weight_decay": tune.choice([0, 1e-6, 1e-8, 1e-10]),
            "project_tag": "lalonde_cps",
            "dag_attention_mask": "True",
            "imbalance_loss_weight": tune.choice([100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0]),
            "prop_score_threshold": tune.choice([0, 0.1]),
            "eps": tune.choice([1e-6, 1e-8, 1e-10]),
            "time_limit": 18000
        },
        "model": {
            "num_layers": tune.choice([2, 4, 8, 16]),
            "dropout_rate": tune.choice([0, 0.1]),
            "embedding_dim": tune.choice([64, 128, 512]),
            "num_heads": tune.choice([4, 8, 16, 32, 64])
        }
    },
  "random_seed": 42
}

config_acic = {
    "filepaths": {
        "dag": "config/dag/acic_dag.json",
        "config": "config/train/acic/acic_sample3.json",
        "data_train_file": "data/acic/sample3/data_train_3.csv",
        "data_val_file": "data/acic/sample3/data_val_3.csv",
        "data_test_file": "data/acic/sample3/data_test_3.csv",
        "pseudo_cate_file": "src/train/acic/aipw_grf_pseudo_cate.csv",
        "result_file_ipw": "experiments/results/acic/acic_best_params_ipw_sample3.json",
        "result_file_cfcv": "experiments/results/acic/acic_best_params_cfcv_sample3.json"
    },
  "cfcv":{
        "training": {
            "num_epochs": tune.choice([10, 20, 30, 40]),
            "batch_size": tune.choice([32, 64, 128, 256, 512]),
            "learning_rate": tune.choice([1e-4, 1e-5, 1e-6]),
            "weight_decay": tune.choice([0, 1e-4, 1e-6, 1e-8]),
            "project_tag": "acic",
            "dag_attention_mask": "True",
            "imbalance_loss_weight": tune.choice([100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0]),
            "prop_score_threshold": tune.choice([0, 0.1]),
            "eps": tune.choice([1e-6, 1e-8, 1e-10]),
            "time_limit": 14400
        },
        "model": {
            "num_layers": tune.choice([2, 4, 8, 16]),
            "dropout_rate": tune.choice([0, 0.1, 0.2, 0.3]),
            "embedding_dim": tune.choice([64, 128, 512]),
            "num_heads": tune.choice([2, 4, 8, 16, 32, 64])
        }
    },
    "ipw":{
            "training": {
            "num_epochs": 100,
            "batch_size": tune.choice([256, 512, 1024]),
            "learning_rate": tune.choice([1e-9]),
            "weight_decay": tune.choice([0, 1e-6, 1e-8, 1e-10]),
            "project_tag": "lalonde_cps",
            "dag_attention_mask": "True",
            "imbalance_loss_weight": tune.choice([100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0]),
            "prop_score_threshold": tune.choice([0, 0.1]),
            "eps": tune.choice([1e-6, 1e-8, 1e-10]),
            "time_limit": 18000
        },
        "model": {
            "num_layers": tune.choice([2, 4, 8, 16]),
            "dropout_rate": tune.choice([0, 0.1]),
            "embedding_dim": tune.choice([64, 128, 512]),
            "num_heads": tune.choice([4, 8, 16, 32, 64])
        }
    },
  "random_seed": 42
}

config_proximal_n1000_u = {
    "data": {
        "name": "demand",
        "n_sample": 1000,
        "random_seed": 42
    },
    "training_transformer": {
        "n_epochs": tune.choice([100, 300, 500, 1000, 3000, 5000]),
        "batch_size": tune.choice([32, 64, 100, 1000]),
        "log_metrics": "False",
        "learning_rate": tune.choice([1e-3, 1e-4, 1e-5]),
        "l2_penalty": tune.choice([1e-3, 1e-4, 1e-5, 1e-6, 3e-06]),
        "alpha": 0,
        "loss_name": "U_statistic",
        "dag_attention_mask": "True"
    },
    "model_transformer": {
        "name": "nmmr",
        "activation": tune.choice(["relu", "gelu"]),
        "network_width": tune.choice([80, 160, 320, 640]),
        "input_layer_depth": tune.choice([4, 8, 16]),
        "num_layers": tune.choice([1, 2]),
        "dropout_rate": tune.choice([0, 0.1, 0.2, 0.3, 0.4]),
        "embedding_dim": tune.choice([10, 20, 40, 80]),
        "feedforward_dim": 80,
        "num_heads": tune.choice([1, 2]),
        "encoder_weight": tune.uniform(0.01, 0.1)
    },
    "n_repeat": 20
}

config_proximal_n1000_u_z = {
    "data": {
        "name": "demand",
        "n_sample": 1000,
        "random_seed": 42
    },
    "training_transformer": {
        "n_epochs": tune.choice([300, 350, 380, 400, 450, 480, 500, 800]),
        "batch_size": tune.choice([32, 64, 100, 1000]),
        "log_metrics": "False",
        "learning_rate": tune.choice([1e-3, 1e-4]),
        "l2_penalty": 3e-06,
        "alpha": 0,
        "loss_name": "U_statistic",
        "dag_attention_mask": "True"
    },
    "model_transformer": {
        "name": "nmmr",
        "activation": tune.choice(["relu", "gelu"]),
        "network_width": tune.choice([80, 160, 320, 640]),
        "input_layer_depth": tune.choice([4, 8, 16]),
        "num_layers": tune.choice([1, 2, 4]),
        "dropout_rate": tune.choice([0, 0.1, 0.2, 0.3]),
        "embedding_dim": tune.choice([10, 20, 40, 80]),
        "feedforward_dim": 80,
        "num_heads": tune.choice([1, 2, 4]),
        "encoder_weight": tune.uniform(0.001, 0.1)
    },
    "n_repeat": 20
}


config_proximal_n5000_u = {
    "data": {
        "name": "demand",
        "n_sample": 5000,
        "random_seed": 42
    },
    "training_transformer": {
        "n_epochs": tune.choice([10, 50, 100, 150, 200]),
        "batch_size": tune.choice([128, 256, 512, 1000, 3000, 5000]),
        "log_metrics": "False",
        "learning_rate": tune.choice([1e-3, 1e-4, 1e-5, 1e-6]),
        "l2_penalty": tune.choice([3, 2, 1, 1e-3, 1e-4, 1e-5, 1e-6]),
        "alpha": tune.uniform(0, 1),
        "loss_name": "U_statistic",
        "dag_attention_mask": "True"
    },
    "model_transformer": {
        "name": "nmmr",
        "num_layers": 1,
        "dropout_rate": tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "embedding_dim": tune.choice([8, 16, 32, 64, 128, 512]),
        "num_heads": tune.grid_search([1, 2, 4])
    },
    "n_repeat": 20
}

config_proximal_n5000_u_z = {
    "data": {
        "name": "demand",
        "n_sample": 5000,
        "random_seed": 42
    },
    "training_transformer": {
        "n_epochs": tune.choice([300, 350, 380, 400, 450, 480, 500, 800, 1000, 2000, 3000]),
        "batch_size": tune.choice([32, 64, 100, 1000]),
        "log_metrics": "False",
        "learning_rate": tune.choice([1e-3, 1e-4]),
        "l2_penalty": tune.choice([3e-04, 3e-05, 3e-06]),
        "alpha": 0,
        "loss_name": "U_statistic",
        "dag_attention_mask": "True"
    },
    "model_transformer": {
        "name": "nmmr",
        "activation": tune.choice(["relu", "gelu"]),
        "network_width": tune.choice([80, 160, 320, 640]),
        "input_layer_depth": tune.choice([4, 8, 16]),
        "num_layers": tune.choice([1, 2, 4]),
        "dropout_rate": tune.choice([0, 0.1, 0.2, 0.3]),
        "embedding_dim": tune.choice([10, 20, 40, 80]),
        "feedforward_dim": 80,
        "num_heads": tune.choice([1, 2, 4]),
        "encoder_weight": tune.uniform(0.001, 0.1)
    },
    "n_repeat": 20
}

config_proximal_n10000_u = {
    "data": {
        "name": "demand",
        "n_sample": 10000,
        "random_seed": 42
    },
    "training_transformer": {
        "n_epochs": tune.choice([10, 50, 100, 150, 200, 300, 500, 1000]),
        "batch_size": tune.choice([256, 512, 1000, 3000, 5000, 7000, 10000]),
        "log_metrics": "False",
        "learning_rate": tune.choice([1e-3, 1e-4, 1e-5, 1e-6]),
        "l2_penalty": tune.choice([3, 2, 1, 1e-3, 1e-4, 1e-5, 1e-6]),
        "alpha": tune.uniform(0, 1),
        "loss_name": "U_statistic",
        "dag_attention_mask": "True"
    },
    "model_transformer": {
        "name": "nmmr",
        "num_layers": 1,
        "dropout_rate": tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "embedding_dim": tune.choice([32, 64, 128, 512]),
        "num_heads": tune.grid_search([1, 2, 4])
    },
    "n_repeat": 20
}


config_proximal_n50000_u = {
    "data": {
        "name": "demand",
        "n_sample": 50000,
        "random_seed": 42
    },
    "training_transformer": {
        "n_epochs": tune.choice([10, 50, 100, 150, 200, 500, 1000]),
        "batch_size": tune.choice([512, 1000, 3000, 5000, 10000, 30000, 50000]),
        "log_metrics": "False",
        "learning_rate": tune.choice([1e-3, 1e-4, 1e-5, 1e-6]),
        "l2_penalty": tune.choice([3, 2, 1, 1e-3, 1e-4, 1e-5, 1e-6]),
        "alpha": tune.uniform(0, 1),
        "loss_name": "U_statistic",
        "dag_attention_mask": "True"
    },
    "model_transformer": {
        "name": "nmmr",
        "num_layers": 1,
        "dropout_rate": tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "embedding_dim": tune.choice([32, 64, 128, 512]),
        "num_heads": tune.grid_search([1, 2, 4])
    },
    "n_repeat": 20
}

config_proximal_n50000_u_z = {
    "data": {
        "name": "demand",
        "n_sample": 50000,
        "random_seed": 42
    },
    "training_transformer": {
        "n_epochs": tune.choice([1000, 2000]),
        "batch_size": tune.choice([32, 64, 100, 1000, 2000]),
        "log_metrics": "False",
        "learning_rate": tune.choice([1e-3, 1e-4]),
        "l2_penalty": tune.choice([3e-04, 3e-05, 3e-06]),
        "alpha": 0,
        "loss_name": "U_statistic",
        "dag_attention_mask": "True"
    },
    "model_transformer": {
        "name": "nmmr",
        "activation": tune.choice(["relu", "gelu"]),
        "network_width": tune.choice([80, 160, 320, 640]),
        "input_layer_depth": tune.choice([4, 8, 16]),
        "num_layers": tune.choice([1, 2, 4]),
        "dropout_rate": tune.choice([0, 0.1, 0.2, 0.3]),
        "embedding_dim": tune.choice([10, 20, 40, 80]),
        "feedforward_dim": 80,
        "num_heads": tune.choice([1, 2, 4]),
        "encoder_weight": tune.uniform(0.001, 0.1)
    },
    "n_repeat": 20
}


config_proximal_n1000_v = {
    "data": {
        "name": "demand",
        "n_sample": 1000,
        "random_seed": 42
    },
    "training_transformer": {
        "n_epochs": tune.choice([10, 50, 100, 150]),
        "batch_size": tune.choice([512, 1000]),
        "log_metrics": "False",
        "learning_rate": tune.choice([1e-3, 1e-4, 1e-5, 1e-6]),
        "l2_penalty": tune.choice([3, 2, 1, 1e-3, 1e-4, 1e-5, 1e-6]),
        "alpha": tune.uniform(0, 1),
        "loss_name": "V_statistic",
        "dag_attention_mask": "True"
    },
    "model_transformer": {
        "name": "nmmr",
        "num_layers": 1,
        "dropout_rate": tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "embedding_dim": tune.choice([8, 16, 32, 64, 128, 512]),
        "num_heads": 1
    },
    "n_repeat": 20
}

config_proximal_n5000_v = {
    "data": {
        "name": "demand",
        "n_sample": 5000,
        "random_seed": 42
    },
    "training_transformer": {
        "n_epochs": tune.choice([10, 50, 100, 150, 200]),
        "batch_size": tune.choice([128, 256, 512, 1000, 3000, 5000]),
        "log_metrics": "False",
        "learning_rate": tune.choice([1e-3, 1e-4, 1e-5, 1e-6]),
        "l2_penalty": tune.choice([3, 2, 1, 1e-3, 1e-4, 1e-5, 1e-6]),
        "alpha": tune.uniform(0, 1),
        "loss_name": "V_statistic",
        "dag_attention_mask": "True"
    },
    "model_transformer": {
        "name": "nmmr",
        "num_layers": 1,
        "dropout_rate": tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "embedding_dim": tune.choice([8, 16, 32, 64, 128, 512]),
        "num_heads": tune.grid_search([1, 2, 4])
    },
    "n_repeat": 20
}

config_proximal_n10000_v = {
    "data": {
        "name": "demand",
        "n_sample": 10000,
        "random_seed": 42
    },
    "training_transformer": {
        "n_epochs": tune.choice([10, 50, 100, 150, 200, 300, 500, 1000]),
        "batch_size": tune.choice([256, 512, 1000, 3000, 5000, 7000, 10000]),
        "log_metrics": "False",
        "learning_rate": tune.choice([1e-3, 1e-4, 1e-5, 1e-6]),
        "l2_penalty": tune.choice([3, 2, 1, 1e-3, 1e-4, 1e-5, 1e-6]),
        "alpha": tune.uniform(0, 1),
        "loss_name": "V_statistic",
        "dag_attention_mask": "True"
    },
    "model_transformer": {
        "name": "nmmr",
        "num_layers": 1,
        "dropout_rate": tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "embedding_dim": tune.choice([32, 64, 128, 512]),
        "num_heads": tune.grid_search([1, 2, 4])
    },
    "n_repeat": 20
}


config_proximal_n50000_v = {
    "data": {
        "name": "demand",
        "n_sample": 50000,
        "random_seed": 42
    },
    "training_transformer": {
        "n_epochs": tune.choice([10, 50, 100, 150, 200, 500, 1000]),
        "batch_size": tune.choice([512, 1000, 3000, 5000, 10000, 30000, 50000]),
        "log_metrics": "False",
        "learning_rate": tune.choice([1e-3, 1e-4, 1e-5, 1e-6]),
        "l2_penalty": tune.choice([3, 2, 1, 1e-3, 1e-4, 1e-5, 1e-6]),
        "alpha": tune.uniform(0, 1),
        "loss_name": "V_statistic",
        "dag_attention_mask": "True"
    },
    "model_transformer": {
        "name": "nmmr",
        "num_layers": 1,
        "dropout_rate": tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "embedding_dim": tune.choice([32, 64, 128, 512]),
        "num_heads": tune.grid_search([1, 2, 4])
    },
    "n_repeat": 20
}
