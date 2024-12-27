from ray.tune import Stopper


class EarlyStoppingAtMinRMSE(Stopper):
    def __init__(self, rmse_key="cfcv_rmse", patience=5):
        self.rmse_key = rmse_key
        self.best_rmse = float("inf")
        self.no_improvement_count = 0
        self.patience = patience

    def __call__(self, trial_id, result):
        current_rmse = result[self.rmse_key]
        if current_rmse < self.best_rmse:
            self.best_rmse = current_rmse
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        return self.no_improvement_count >= self.patience

    def stop_all(self):
        return False