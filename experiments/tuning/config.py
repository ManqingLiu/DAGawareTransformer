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