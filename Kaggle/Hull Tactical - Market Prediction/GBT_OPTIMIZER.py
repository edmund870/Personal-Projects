import os
from dataclasses import dataclass, field
from loguru import logger
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from CONFIG import CONFIG
from GBT_TRAINER import LightGBMTrainer, XGBoostTrainer


@dataclass
class OptunaTuner:
    X: np.ndarray
    y: np.ndarray
    final_x_test: np.ndarray
    risk_free_rate: np.ndarray
    final_risk_free_rate: np.ndarray
    forward_returns: np.ndarray
    final_forward_returns: np.ndarray
    feature_names: List
    n_trials: int = 30
    best_score = float("-inf")
    best_params = None

    def objective_xgb(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization

        Returns:
            Average Sharpe ratio across all folds
        """

        # Suggest XGBoost hyperparameters
        params = {
            "eta": trial.suggest_float("eta", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "lambda": trial.suggest_float("lambda", 0.1, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 0.1, 1.0, log=True),
            "gamma": trial.suggest_float("gamma", 0, 3),
            "tree_method": "hist",
            "seed": CONFIG.RANDOM_STATE,
            "disable_default_eval_metric": 1,
            "verbosity": 0,
        }

        # Suggest incremental learning hyperparameters
        batch_size = trial.suggest_int("batch_size", 100, 500, step=10)
        num_boost_round = trial.suggest_int("num_boost_round", 100, CONFIG.NUM_BOOST_ROUND, step=100)

        # Store for later access
        trial.set_user_attr("batch_size", batch_size)
        trial.set_user_attr("num_boost_round", num_boost_round)

        xgb_trainer = XGBoostTrainer(params=params)
        xgb_res = xgb_trainer.train_cv(
            X=self.X,
            y=self.y,
            final_x_test=self.final_x_test,
            risk_free_rate=self.risk_free_rate,
            forward_returns=self.forward_returns,
            final_risk_free_rate=self.final_risk_free_rate,
            final_forward_returns=self.final_forward_returns,
            batch_size=batch_size,
            NUM_BOOST_ROUND=num_boost_round,
        )

        # Return average score across folds
        avg_score = xgb_res[-1]

        # Report intermediate value for pruning
        trial.report(avg_score, step=0)

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

        return avg_score

    def objective_lgb(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization

        Returns:
            Average Sharpe ratio across all folds
        """
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 60), 
            "max_depth": trial.suggest_int("max_depth", 4, 8), 
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 15, 50),
            "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-3, 10.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 0.8), 
            "feature_fraction_bynode": trial.suggest_float("feature_fraction_bynode", 0.4, 0.8),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.9),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.01, 10.0, log=True), 
            "lambda_l2": trial.suggest_float("lambda_l2", 0.01, 10.0, log=True),  
            "max_bin": trial.suggest_int("max_bin", 63, 255),  
            "metric": "None", 
            "verbosity": -1,
            "seed": CONFIG.RANDOM_STATE,
        }

        # Suggest incremental learning hyperparameters
        batch_size = trial.suggest_int("batch_size", 100, 500, step=10)
        num_boost_round = trial.suggest_int("num_boost_round", 100, CONFIG.NUM_BOOST_ROUND, step=100)

        # Store for later access
        trial.set_user_attr("batch_size", batch_size)
        trial.set_user_attr("num_boost_round", num_boost_round)

        lgb_trainer = LightGBMTrainer(params=params)
        lgb_res = lgb_trainer.train_cv(
            X=self.X,
            y=self.y,
            final_x_test=self.final_x_test,
            risk_free_rate=self.risk_free_rate,
            forward_returns=self.forward_returns,
            final_risk_free_rate=self.final_risk_free_rate,
            final_forward_returns=self.final_forward_returns,
            batch_size=batch_size,
            NUM_BOOST_ROUND=num_boost_round,
        )

        # Return average score across folds
        avg_score = lgb_res[-1]

        # Report intermediate value for pruning
        trial.report(avg_score, step=0)

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

        return avg_score

    def optimize(self, study_name, folder, model, n_jobs: int = 1):
        """
        Run Optuna optimization

        Args:
            n_jobs: Number of parallel jobs
            timeout: Timeout in seconds
        """
        os.makedirs(folder, exist_ok=True)
        # Create study with TPE sampler and Median pruner
        sampler = TPESampler(n_startup_trials=10, seed=CONFIG.RANDOM_STATE)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)

        study = optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{folder}/{study_name}.db",
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        # Run optimization
        study.optimize(
            self.objective_xgb if model == "XGB" else self.objective_lgb,
            n_trials=self.n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
        )

        # Get best trial
        best_trial = study.best_trial

        logger.info(f"Best trial value: {best_trial.value:.4f}")
        logger.info("Best trial params:")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")

        self.best_score = best_trial.value
        self.best_params = best_trial.params

        return study
