from ray import tune

config_psid = {
  "filepaths": {
    "dag": "config/dag/lalonde_psid_dag.json",
    "config": "config/train/lalonde_psid.json",
    "data_train_file": "data/lalonde/ldw_psid/lalonde_psid_train.csv",
    "data_val_file": "data/lalonde/ldw_psid/lalonde_psid_val.csv",
    "data_test_file": "data/lalonde/ldw_psid/lalonde_psid_test.csv",
    "output_file": "experiments/predict/lalonde_psid/predictions_psid.csv",
    "result_file": "config/train/lalonde_psid/lalonde_psid_best_params.json"
  },
  "training": {
    "num_epochs": tune.choice([20, 30, 40, 50]),
    "batch_size": tune.choice([32, 64, 128]),
    "learning_rate": tune.choice([1e-5, 1e-6, 1e-7]),
    "weight_decay": tune.choice([0, 1e-3, 1e-4]),
    "project_tag": "lalonde_psid",
    "dag_attention_mask": tune.choice([True]),
    "imbalance_loss_weight": tune.choice([10, 5, 1, 0.5, 0.1, 0.05, 0.01])
  },
  "model": {
    "num_layers": tune.choice([2, 4, 8]),
    "dropout_rate": tune.choice([0, 0.1, 0.2]),
    "embedding_dim": tune.choice([64, 128, 512]),
    "num_heads": tune.choice([2, 4, 8, 16, 32, 64])
  },
  "random_seed": tune.choice([42])
}

config_cps = {
  "training": {
    "num_epochs": tune.choice([100]),
    "batch_size": tune.choice([128, 256, 512]),
    "learning_rate": tune.choice([1e-3, 1e-4, 1e-5]),
    "weight_decay": tune.choice([0, 0.0001, 0.0002]),
    "project_tag": "lalonde_cps"
  },
  "model": {
    "num_layers": tune.choice([3, 4, 5]),
    "dropout_rate": tune.choice([0, 0.1, 0.2, 0.3]),
    "embedding_dim": tune.choice([128, 256, 512]),
    "num_heads": tune.choice([16, 32, 64])
  }
}

config_twins = {
  "training": {
    "num_epochs": tune.choice([100]),
    "batch_size": tune.choice([128, 256, 512]),
    "learning_rate": tune.choice([1e-3, 1e-4, 1e-5]),
    "weight_decay": tune.choice([0, 0.0001, 0.0002]),
    "project_tag": "twins"
  },
  "model": {
    "num_layers": tune.choice([3, 4, 5]),
    "dropout_rate": tune.choice([0, 0.1, 0.2, 0.3]),
    "embedding_dim": tune.choice([128, 256, 512]),
    "num_heads": tune.choice([16, 32, 64])
  }
}