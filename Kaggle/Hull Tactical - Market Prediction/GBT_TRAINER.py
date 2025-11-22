from dataclasses import dataclass, field
import pickle
import os
from typing import Callable, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb
from loguru import logger
from numba import njit
from sklearn.model_selection import TimeSeriesSplit


from CONFIG import CONFIG
from GBT_METRICS import CustomMetrics
from OPTIMAL_SOLUTION_SOLVER import optimize


@dataclass
class TimeSeriesModelTrainer:
    """Base class for time series model training with cross-validation"""

    cv_results: List[Dict] = field(default_factory=list)
    models: List = field(default_factory=list)
    shap_df: Optional[pl.DataFrame] = None
    feature_names: Optional[List[str]] = None

    def train_cv(self, X: np.ndarray, y: np.ndarray, risk_free_rate: np.ndarray, forward_returns: np.ndarray) -> pl.DataFrame:
        """
        Perform time series cross-validation training.

        Args:
            X: Feature matrix
            y: Target variable
            risk_free_rate: Risk-free rate array
            forward_returns: Forward returns array
            dates: Optional datetime index for logging

        Returns:
            DataFrame with cross-validation results
        """
        raise NotImplementedError("Subclasses must implement train_cv method")

    def get_results_summary(self) -> pl.DataFrame:
        """Get summary of cross-validation results"""
        df = pl.DataFrame(self.cv_results)
        logger.info(f"Mean Test Sharpe: {df['final_score'].mean():.4f} (+/- {df['final_score'].std():.4f})")

        return df


class XGBoostTrainer(TimeSeriesModelTrainer):
    """XGBoost training pipeline with time series cross-validation"""

    def __init__(self, params: Optional[Dict] = None):
        super().__init__()
        self.params = params or self._default_params()

    def _default_params(self) -> Dict:
        """Default XGBoost parameters"""
        return {"max_depth": 6, "eta": 0.1, "device": "cuda", "tree_method": "hist", "seed": CONFIG.RANDOM_STATE, "disable_default_eval_metric": 1}

    def train_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        final_x_test: np.ndarray | None,
        risk_free_rate: np.ndarray,
        forward_returns: np.ndarray,
        final_risk_free_rate: np.ndarray | None,
        final_forward_returns: np.ndarray | None,
        batch_size: int,
        NUM_BOOST_ROUND: int,
        folder=None,
    ) -> Tuple[pl.DataFrame, float]:
        """Train XGBoost with time series cross-validation"""

        tscv = TimeSeriesSplit(n_splits=CONFIG.N_FOLDS)

        for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
            # Split data
            X_train, X_test = X[train_index], X[max(train_index) :]
            y_train, y_test = y[train_index], y[max(train_index) :]

            rfr_train, rfr_test = risk_free_rate[train_index], risk_free_rate[max(train_index) :]
            fwd_train, fwd_test = forward_returns[train_index], forward_returns[max(train_index) :]

            split_idx = int(len(X_train) * 0.8)
            X_train, X_val = X_train[:split_idx], X_train[split_idx:]
            y_train, y_val = y_train[:split_idx], y_train[split_idx:]
            rfr_train, rfr_val = rfr_train[:split_idx], rfr_train[split_idx:]
            fwd_train, fwd_val = fwd_train[:split_idx], fwd_train[split_idx:]

            # Create DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            # Create custom metrics
            train_metric = CustomMetrics.create_volatility_adjusted_sharpe_xgb(rfr_train, fwd_train)
            val_metric = CustomMetrics.create_volatility_adjusted_sharpe_xgb(rfr_val, fwd_val)

            # Train
            evals_result = {}
            bst = xgb.train(
                self.params,
                dtrain,
                num_boost_round=NUM_BOOST_ROUND,
                evals=[(dval, "val")],
                custom_metric=val_metric,
                evals_result=evals_result,
                early_stopping_rounds=CONFIG.EARLY_STOPPING_ROUNDS,
                maximize=True,
                verbose_eval=CONFIG.VERBOSE_EVAL,
            )

            BATCH_SIZE = batch_size
            bst_incremental = bst
            test_predictions = []
            final_test_predictions = []

            logger.info("INCREMENTAL LEARNING STARTED")
            for batch_start in range(0, len(X_test), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(X_test))

                # Extract batch data
                X_batch = X_test[batch_start:batch_end]
                rfr_batch = rfr_test[batch_start:batch_end]
                fwd_batch = fwd_test[batch_start:batch_end]

                # compute optimal solution
                y_batch = optimize(train=rfr_batch, rfr=rfr_batch, fwd=fwd_batch)

                # Make predictions with current model before updating
                d_batch_pred = xgb.DMatrix(X_batch)
                batch_preds = bst_incremental.predict(d_batch_pred)
                test_predictions.extend(batch_preds)

                # Create DMatrix for incremental training
                d_batch_train = xgb.DMatrix(X_batch, label=y_batch)
                batch_metric = CustomMetrics.create_volatility_adjusted_sharpe_xgb(rfr_batch, fwd_batch)

                # Update model with new batch
                bst_incremental = xgb.train(
                    self.params,
                    d_batch_train,
                    num_boost_round=int(NUM_BOOST_ROUND / 5),
                    xgb_model=bst_incremental,
                    custom_metric=batch_metric,
                    maximize=True,
                    verbose_eval=False,
                )

            if folder:
                os.makedirs(folder, exist_ok=True)
                file_name = f"{folder}/xgb_{fold}.pkl"
                pickle.dump(bst, open(file_name, "wb"))

            final_score = CustomMetrics.comp_metric(predt=np.array(test_predictions), rfr_data=rfr_test, fwd_data=fwd_test)

            # Store results
            self.models.append(bst)

            self.cv_results.append(
                {
                    "fold": fold,
                    "model": "XGBoost",
                    "train_size": len(train_index),
                    "test_size": len(X_test),
                    "best_iteration": bst.best_iteration,
                    "best_score": bst.best_score,
                    "train_start": train_index[0],
                    "train_end": train_index[-1],
                    "test_start": max(train_index),
                    "final_score": final_score[-1],
                }
            )

        if final_x_test is not None:
            logger.info("FINAL_TEST STARTED")
            bst_incremental = self.models
            for batch_start in range(0, len(final_x_test), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(final_x_test))

                # Extract batch data
                X_batch = final_x_test[batch_start:batch_end]
                rfr_batch = final_risk_free_rate[batch_start:batch_end]  # type:ignore
                fwd_batch = final_forward_returns[batch_start:batch_end]  # type:ignore

                # compute optimal solution
                y_batch = optimize(train=rfr_batch, rfr=rfr_batch, fwd=fwd_batch)

                # Make predictions with current model before updating
                d_batch_pred = xgb.DMatrix(X_batch)
                batch_preds = [model.predict(d_batch_pred) for model in bst_incremental]
                final_test_predictions.extend(np.mean(batch_preds, axis=0))

                # Create DMatrix for incremental training
                d_batch_train = xgb.DMatrix(X_batch, label=y_batch)
                batch_metric = CustomMetrics.create_volatility_adjusted_sharpe_xgb(rfr_batch, fwd_batch)

                # Update model with new batch
                bst_incremental = [
                    xgb.train(
                        self.params,
                        d_batch_train,
                        num_boost_round=int(NUM_BOOST_ROUND / 5),
                        xgb_model=model,
                        custom_metric=batch_metric,
                        maximize=True,
                        verbose_eval=False,
                    )
                    for model in bst_incremental
                ]

            final_test_score = CustomMetrics.comp_metric(
                predt=np.array(final_test_predictions),
                rfr_data=final_risk_free_rate,  # type:ignore
                fwd_data=final_forward_returns,  # type:ignore
            )
        else:
            final_test_score = 0

        avg = np.mean([cv.get("final_score") for cv in self.cv_results])  # type:ignore
        logger.info(f"Avg: {avg} | final test score: {final_test_score}")

        return self.get_results_summary(), final_test_score[-1]  # type:ignore


class LightGBMTrainer(TimeSeriesModelTrainer):
    """LightGBM training pipeline with time series cross-validation"""

    def __init__(self, params: Optional[Dict] = None):
        super().__init__()
        self.params = params or self._default_params()

    def _default_params(self) -> Dict:
        """Default LightGBM parameters"""
        return {"max_depth": 6, "learning_rate": 0.1, "seed": CONFIG.RANDOM_STATE, "verbose": -1, "metric": "None"}

    def train_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        final_x_test: np.ndarray | None,
        risk_free_rate: np.ndarray,
        forward_returns: np.ndarray,
        final_risk_free_rate: np.ndarray | None,
        final_forward_returns: np.ndarray | None,
        batch_size: int,
        NUM_BOOST_ROUND: int,
        folder=None,
    ) -> Tuple[pl.DataFrame, float]:
        """Train LightGBM with time series cross-validation"""

        tscv = TimeSeriesSplit(n_splits=CONFIG.N_FOLDS)

        for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
            # Split data
            X_train, X_test = X[train_index], X[max(train_index) :]
            y_train, y_test = y[train_index], y[max(train_index) :]

            # Split risk_free_rate and forward_returns for train/test
            rfr_train, rfr_test = risk_free_rate[train_index], risk_free_rate[max(train_index) :]
            fwd_train, fwd_test = forward_returns[train_index], forward_returns[max(train_index) :]

            # Further split train into train/val (including rfr and fwd)
            split_idx = int(len(X_train) * 0.8)
            X_train, X_val = X_train[:split_idx], X_train[split_idx:]
            y_train, y_val = y_train[:split_idx], y_train[split_idx:]
            rfr_train, rfr_val = rfr_train[:split_idx], rfr_train[split_idx:]
            fwd_train, fwd_val = fwd_train[:split_idx], fwd_train[split_idx:]

            # Create Datasets
            train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)

            # Create custom metric for validation
            val_metric = CustomMetrics.create_volatility_adjusted_sharpe_lgb(rfr_val, fwd_val)

            # Initial training with validation set
            evals_result = {}
            bst = lgb.train(
                self.params,
                train_data,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[val_data],
                valid_names=["validation"],
                feval=val_metric,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=CONFIG.EARLY_STOPPING_ROUNDS, verbose=True),
                    lgb.log_evaluation(period=CONFIG.VERBOSE_EVAL),
                    lgb.record_evaluation(evals_result),
                ],
            )

            # Get best iteration from initial training
            best_iteration = bst.best_iteration
            best_score = bst.best_score["validation"]["adj_sharpe"]

            # Incremental learning on test set with batch updates
            BATCH_SIZE = batch_size  # Configure based on your timestep requirements
            bst_incremental = bst
            test_predictions = []
            final_test_predictions = []

            logger.info("INCREMENTAL LEARNING STARTED")
            for batch_start in range(0, len(X_test), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(X_test))

                # Extract batch data
                X_batch = X_test[batch_start:batch_end]
                rfr_batch = rfr_test[batch_start:batch_end]
                fwd_batch = fwd_test[batch_start:batch_end]

                # compute optimal solution
                y_batch = optimize(train=rfr_batch, rfr=rfr_batch, fwd=fwd_batch)

                # Make predictions with current model before updating
                batch_preds = bst_incremental.predict(X_batch)
                test_predictions.extend(batch_preds)  # type:ignore

                # Create Dataset for incremental training
                batch_data = lgb.Dataset(X_batch, label=y_batch, reference=train_data, free_raw_data=False)
                batch_metric = CustomMetrics.create_volatility_adjusted_sharpe_lgb(rfr_batch, fwd_batch)

                # Update model with new batch using init_model
                bst_incremental = lgb.train(
                    self.params,
                    batch_data,
                    num_boost_round=int(NUM_BOOST_ROUND / 5),  # Add new trees per batch (adjust as needed)
                    init_model=bst_incremental,  # Continue from previous model
                    valid_sets=[batch_data],
                    valid_names=["batch"],
                    feval=batch_metric,
                    callbacks=[
                        lgb.log_evaluation(period=0),  # Silent during batch updates
                    ],
                )

                if folder:
                    os.makedirs(folder, exist_ok=True)
                    file_name = f"{folder}/lgb_{fold}.pkl"
                    pickle.dump(bst, open(file_name, "wb"))
            final_score = CustomMetrics.comp_metric(predt=np.array(test_predictions), rfr_data=rfr_test, fwd_data=fwd_test)

            # Store results
            self.models.append(bst)
            best_iteration = bst.best_iteration
            # best_score = bst.best_score["test"]["adj_sharpe"]

            self.cv_results.append(
                {
                    "fold": fold,
                    "model": "LightGBM",
                    "train_size": len(train_index),
                    "test_size": len(X_test),
                    "best_iteration": best_iteration,
                    "best_score": best_score,
                    "train_start": train_index[0],
                    "train_end": train_index[-1],
                    "test_start": max(train_index),
                    "final_score": final_score[-1],
                }
            )

        bst_incremental = self.models
        if final_x_test is not None:
            logger.info("FINAL_TEST STARTED")
            for batch_start in range(0, len(final_x_test), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(final_x_test))

                # Extract batch data
                X_batch = final_x_test[batch_start:batch_end]
                rfr_batch = final_risk_free_rate[batch_start:batch_end]  # type:ignore
                fwd_batch = final_forward_returns[batch_start:batch_end]  # type:ignore

                # compute optimal solution
                y_batch = optimize(train=rfr_batch, rfr=rfr_batch, fwd=fwd_batch)

                # Make predictions with current model before updating
                batch_preds = [model.predict(X_batch) for model in bst_incremental]
                final_test_predictions.extend(np.mean(batch_preds, axis=0))  # type:ignore

                # Create Dataset for incremental training
                batch_data = lgb.Dataset(X_batch, label=y_batch, free_raw_data=False)
                batch_metric = CustomMetrics.create_volatility_adjusted_sharpe_lgb(rfr_batch, fwd_batch)

                # Update model with new batch using init_model
                bst_incremental = [
                    lgb.train(
                        self.params,
                        batch_data,
                        num_boost_round=int(NUM_BOOST_ROUND / 5),  # Add new trees per batch (adjust as needed)
                        init_model=model,  # Continue from previous model
                        valid_sets=[batch_data],
                        valid_names=["batch"],
                        feval=batch_metric,
                        callbacks=[
                            lgb.log_evaluation(period=0),  # Silent during batch updates
                        ],
                    )
                    for model in bst_incremental
                ]

            final_test_score = CustomMetrics.comp_metric(
                predt=np.array(final_test_predictions),
                rfr_data=final_risk_free_rate,  # type:ignore
                fwd_data=final_forward_returns,  # type:ignore
            )
        else:
            final_test_score = 0

        avg = np.mean([cv.get("final_score") for cv in self.cv_results])  # type:ignore
        logger.info(f"Avg: {avg} | final test avg: {final_test_score}")

        return self.get_results_summary(), final_test_score[-1]  # type:ignore
