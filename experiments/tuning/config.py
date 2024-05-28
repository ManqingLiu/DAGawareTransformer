from ray import tune

config_psid = {
  "training": {
    "num_epochs": tune.choice([10]),
    "batch_size": tune.choice([32, 64, 128]),
    "learning_rate": tune.choice([1e-3, 1e-4]),
    "weight_decay": tune.choice([0, 0.01]),
    "project_tag": "lalonde_psid"
  },
  "model": {
    "num_layers": tune.choice([2, 3, 4]),
    "dropout_rate": tune.choice([0, 0.1]),
    "embedding_dim": tune.choice([128, 256, 512]),
    "num_heads": tune.choice([16, 32, 64])
  }
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