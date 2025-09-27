from typing import Dict, List, Optional
import copy
import numpy as np
from scipy.stats import rankdata
import random

import torch
from torch.utils.data import Dataset, DataLoader


from CONFIG import CONFIG
from DATASET import SequentialDataset
from ENSEMBLE_NN import ENSEMBLE_NN
from LOSS_V2 import CombinedICIRLoss
import time


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result

    return wrapper


class NN:
    """
    Wrapper class for training, validating, updating, and predicting using an ensemble PyTorch model.

    Parameters
    ----------
    model : ENSEMBLE_NN
        PyTorch model to be trained and evaluated.
    lr : float, optional
        Initial learning rate for optimizer (default 0.001).
    batch_size : int, optional
        Mini-batch size for training (default 1).
    epochs : int, optional
        Maximum number of training epochs (default 100).
    early_stopping_patience : int, optional
        Number of epochs with no improvement before stopping (default 10).
    early_stopping : bool, optional
        Whether to use early stopping based on validation metric (default True).
    lr_patience : int, optional
        Number of epochs to wait before reducing learning rate (default 2).
    lr_factor : float, optional
        Factor by which to reduce learning rate (default 0.5).
    lr_refit : float, optional
        Learning rate for online refit updates (default 0.001).
    random_seed : int, optional
        Seed for reproducibility (default CONFIG.RANDOM_STATE).
    refit : bool, optional
        Whether to allow incremental refit during validation (default True).
    **kwargs : dict
        Additional keyword arguments.

    Attributes
    ----------
    device : torch.device
        Device used for computation (CPU or GPU).
    criterion : CombinedICIRLoss
        Custom loss function for IC/ICIR optimization.
    optimizer : torch.optim.Optimizer
        Optimizer for model training.
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler.
    refit_optimizer : torch.optim.Optimizer
        Optimizer used for incremental updates.
    best_epoch : int or None
        Epoch number corresponding to best validation performance.
    features : Any
        Placeholder for feature metadata.
    kwargs : dict
        Additional keyword arguments passed to the class.
    """

    def __init__(
        self,
        model: ENSEMBLE_NN,
        lr: float = 0.001,
        batch_size: int = CONFIG.BATCH_SIZE,
        seq_len: int = CONFIG.SEQ_LEN,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        early_stopping: bool = True,
        lr_patience: int = 2,
        lr_factor: float = 0.5,
        lr_refit: float = 0.001,
        spearman_weight: float = 0.4,
        listnet_weight: float = 0.2,
        kendall_weight: float = 0.15,
        pairwise_weight: float = 0.1,
        topk_weight: float = 0.1,
        mse_weight: float = 0.1,  # Small MSE for stability
        listnet_temp: float = 1.0,
        kendall_temp: float = 0.1,
        random_seed: int = CONFIG.RANDOM_STATE,
        refit: bool = True,
        **kwargs,
    ) -> None:
        self.lr = lr
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping = early_stopping
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_refit = lr_refit
        self.random_seed = random_seed
        self.refit = refit

        self.criterion = CombinedICIRLoss(
            spearman_weight=spearman_weight,
            listnet_weight=listnet_weight,
            kendall_weight=kendall_weight,
            pairwise_weight=pairwise_weight,
            topk_weight=topk_weight,
            mse_weight=mse_weight,  # Small MSE for stability
            listnet_temp=listnet_temp,
            kendall_temp=kendall_temp,
        )
        self.val_criterion = CombinedICIRLoss(
            spearman_weight=spearman_weight,
            listnet_weight=listnet_weight,
            kendall_weight=kendall_weight,
            pairwise_weight=pairwise_weight,
            topk_weight=topk_weight,
            mse_weight=mse_weight,  # Small MSE for stability
            listnet_temp=listnet_temp,
            kendall_temp=kendall_temp,
        )
        self.update_criterion = CombinedICIRLoss(
            spearman_weight=spearman_weight,
            listnet_weight=listnet_weight,
            kendall_weight=kendall_weight,
            pairwise_weight=pairwise_weight,
            topk_weight=topk_weight,
            mse_weight=mse_weight,  # Small MSE for stability
            listnet_temp=listnet_temp,
            kendall_temp=kendall_temp,
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)

        self.best_epoch = None
        self.features = None
        self.kwargs = kwargs

    def rank_correlation_sharpe(self, targets, predictions) -> float:
        """
        Compute rank correlation (Spearman) Sharpe ratio between predictions and targets.

        Parameters
        ----------
        targets : np.ndarray
            Array of true target values with shape (n_samples, n_assets).
        predictions : np.ndarray
            Array of model predictions with shape (n_samples, n_assets).

        Returns
        -------
        float
            Sharpe ratio (mean / std) of rank correlations across samples.

        Raises
        ------
        ZeroDivisionError
            If any row or overall standard deviation is zero.
        """
        correlations = []

        for i, (pred_row, target_row) in enumerate(zip(predictions, targets)):
            # Find valid (non-NaN) assets for this timestep
            valid_mask = ~np.isnan(target_row)
            valid_pred = pred_row[valid_mask]
            valid_target = target_row[valid_mask]

            if np.std(pred_row) == 0 or np.std(target_row) == 0:
                raise ZeroDivisionError("Zero standard deviation in a row.")

            rho = np.corrcoef(rankdata(valid_pred, method="average"), rankdata(valid_target, method="average"))[0, 1]
            correlations.append(rho)

        daily_rank_corrs = np.array(correlations)
        std_dev = daily_rank_corrs.std(ddof=0)
        if std_dev == 0:
            raise ZeroDivisionError("Denominator is zero, unable to compute Sharpe ratio.")

        sharpe_ratio = daily_rank_corrs.mean() / std_dev
        return float(sharpe_ratio)

    def flatten_collate_fn(self, batch: list) -> dict[str, torch.Tensor]:
        """
        Collate function for DataLoader to stack sequences and targets.

        Parameters
        ----------
        batch : list
            List of tuples (X_seq, y_current) containing tensors for each sample.

        Returns
        -------
        dict
            Dictionary with keys:
            'continuous' : torch.Tensor
                Stacked input sequences (batch_size, seq_len, features).
            'current' : torch.Tensor
                Stacked target values (batch_size, output_dim).
        """
        continuous_batch, curr_y = zip(*batch)

        # Stack each lag sequence across the batch
        continuous_batch = torch.stack(continuous_batch)
        curr_y = torch.stack(curr_y)

        return {"continuous": continuous_batch, "current": curr_y}

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def fit(self, train_set: tuple, val_set: tuple, retrain_set: tuple, verbose: bool = False) -> None:
        """
        Train the model using training and validation sets with optional early stopping.

        Parameters
        ----------
        train_set : tuple
            Tuple of (X_train, y_train) for sequential training.
        val_set : tuple
            Tuple of (X_val, y_val) for validation monitoring.
        retrain_set : tuple
            Tuple of (X_retrain, y_retrain) used for incremental updates during validation.
        verbose : bool, optional
            If True, prints epoch-level training metrics (default False).
        """
        torch.manual_seed(self.random_seed)
        g = torch.Generator()
        g.manual_seed(CONFIG.RANDOM_STATE)

        seq_train_dataset = SequentialDataset(*train_set, seq_len=self.seq_len)
        seq_train_dataloader = DataLoader(
            seq_train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.flatten_collate_fn,
            worker_init_fn=self.seed_worker,
            generator=g,
            # pin_memory=True,
            # num_workers=2,
            # persistent_workers=True,
            # prefetch_factor=2,
            drop_last=True,
        )

        seq_val_dataset = SequentialDataset(*val_set, seq_len=self.seq_len)
        seq_val_dataloader = DataLoader(
            seq_val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.flatten_collate_fn,
            worker_init_fn=self.seed_worker,
            generator=g,
            # pin_memory=True,
            # num_workers=6,
            # persistent_workers=True,
            # prefetch_factor=2,
            drop_last=True,
        )

        retrain_val_dataset = SequentialDataset(*retrain_set, seq_len=self.seq_len)
        retrain_val_dataloader = DataLoader(
            retrain_val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.flatten_collate_fn,
            worker_init_fn=self.seed_worker,
            generator=g,
            # pin_memory=True,
            # num_workers=6,
            # persistent_workers=True,
            # prefetch_factor=2,
            drop_last=True,
        )

        train_sharpes, val_sharpes = [], []
        if verbose:
            print(f"Device: {self.device}")
            print(
                f"{'Epoch':^7} | "
                f"{'TrainLoss':^10} | {'ValLoss':^10} | "
                f"{'TrainSharpe':^12} | {'ValSharpe':^10} | "
                f"{'TrainICIR':^10} | {'ValICIR':^10} | "
                f"{'TrainListNet':^13} | {'ValListNet':^13} | "
                f"{'TrainKendall':^13} | {'ValKendall':^13} | "
                f"{'TrainPairwise':^14} | {'ValPairwise':^14} | "
                f"{'TrainTopK':^11} | {'ValTopK':^11} | "
                f"{'TrainMSE':^10} | {'ValMSE':^10} | "
                f"{'Train sharpe':^9} | {'Val sharpe':^7} | {'LR':^7}"
            )
            print("-" * 60)

        min_val_sharpe = -np.inf
        best_epoch = 0
        no_improvement = 0
        best_model = None
        for epoch in range(self.epochs):
            train_loss, train_sharpe, train_icir_loss, train_list_net, train_kendall, train_pairwise, train_topk, train_mse = self.train_one_epoch(
                seq_train_dataloader, verbose
            )
            (
                val_loss,
                val_sharpe,
                val_icir_loss,
                val_list_net_loss,
                val_kendall_loss,
                val_pairwise_loss,
                val_topk_loss,
                val_mse_loss,
            ) = self.validate_one_epoch(seq_val_dataloader, retrain_val_dataloader, verbose)

            lr_last = self.optimizer.param_groups[0]["lr"]

            train_sharpes.append(train_sharpe)
            val_sharpes.append(val_sharpe)

            if verbose:
                print(
                    f"{epoch + 1:^7} | "
                    f"{train_loss:^10.4f} | {val_loss:^10.4f} | "
                    f"{train_icir_loss:^10.4f} | {val_icir_loss:^10.4f} | "
                    f"{train_list_net:^13.4f} | {val_list_net_loss:^13.4f} | "
                    f"{train_kendall:^13.4f} | {val_kendall_loss:^13.4f} | "
                    f"{train_pairwise:^14.4f} | {val_pairwise_loss:^14.4f} | "
                    f"{train_topk:^11.4f} | {val_topk_loss:^11.4f} | "
                    f"{train_mse:^10.4f} | {val_mse_loss:^10.4f} | "
                    f"{train_sharpe:^12.4f} | {val_sharpe:^10.4f} | {lr_last:^7.5f}"
                )

            if val_sharpe > min_val_sharpe:
                min_val_sharpe = val_sharpe
                best_model = copy.deepcopy(self.model.state_dict())
                no_improvement = 0
                best_epoch = epoch
            else:
                no_improvement += 1

            if self.early_stopping:
                if no_improvement >= self.early_stopping_patience + 1:
                    self.best_epoch = best_epoch + 1
                    if verbose:
                        print(f"Early stopping on epoch {best_epoch + 1}. Best score: {min_val_sharpe:.4f}")
                    break

        # Load the best model
        if self.early_stopping:
            self.model.load_state_dict(best_model)

    def train_one_epoch(self, seq_train_dataloader: DataLoader, verbose: bool) -> tuple:
        """
        Train the model for a single epoch.

        Parameters
        ----------
        seq_train_dataloader : DataLoader
            DataLoader yielding batches from the training set.
        verbose : bool
            If True, prints batch-level info (not used here).

        Returns
        -------
        tuple
            Tuple containing:
            - train_loss : float, average total loss over batches
            - train_sharpe : float, Spearman Sharpe ratio
            - train_icir_loss : float, average ICIR loss
            - train_mse_loss : float, average MSE loss
            - train_huber_loss : float, average Huber Loss
        """
        self.model.train()
        total_loss = 0.0
        total_icir_loss = 0.0
        total_list_net = 0.0
        total_kendall = 0.0
        total_pairwise = 0.0
        total_topk = 0.0
        total_mse = 0.0

        y_total, preds_total = [], []

        for seq_batch in seq_train_dataloader:
            seq_x_batch = seq_batch["continuous"]

            true_y = seq_batch["current"]

            self.optimizer.zero_grad()
            with torch.autocast(device_type="cuda"):
                pred_y = self.model(seq_x_batch)
                loss, loss_components = self.criterion(pred_y, true_y)
                icir_loss, list_net, kendall, pairwise, topk, mse = loss_components.values()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            total_icir_loss += icir_loss.item()
            total_list_net += list_net.item()
            total_kendall += kendall.item()
            total_pairwise += pairwise.item()
            total_topk += topk.item()
            total_mse += mse.item()

            y_total.append(true_y)
            preds_total.append(pred_y.detach())

            torch.cuda.empty_cache()

        y_total = torch.cat(y_total).cpu().numpy().astype(np.float64)
        preds_total = torch.cat(preds_total).cpu().numpy().astype(np.float64)

        train_sharpe = self.rank_correlation_sharpe(y_total, preds_total)
        len_data_loader = len(seq_train_dataloader)
        train_loss = total_loss / len_data_loader
        train_icir_loss = total_icir_loss / len_data_loader
        train_list_net = total_list_net / len_data_loader
        train_kendall = total_kendall / len_data_loader
        train_pairwise = total_pairwise / len_data_loader
        train_topk = total_topk / len_data_loader
        train_mse = total_mse / len_data_loader

        return train_loss, train_sharpe, train_icir_loss, train_list_net, train_kendall, train_pairwise, train_topk, train_mse

    @timer
    def validate_one_epoch(self, seq_val_dataloader: DataLoader, retrain_val_dataloader: DataLoader, verbose=False) -> tuple:
        """
        Evaluate the model on validation data, optionally refitting incrementally.

        Parameters
        ----------
        seq_val_dataloader : DataLoader
            DataLoader for validation sequences.
        retrain_val_dataloader : DataLoader
            DataLoader for incremental refit sequences.
        verbose : bool, optional
            Whether to print progress (default False).

        Returns
        -------
        tuple
            Tuple containing:
            - val_loss : float, mean validation loss
            - val_sharpe : float, Spearman Sharpe on validation set
            - val_icir_loss : float, mean ICIR loss
            - val_mse_loss : float, mean MSE loss
            - val_huber_loss : float, mean Huber Loss
        """
        model = copy.deepcopy(self.model)

        losses, icir_losses, list_net_losses, kendall_losses, pairwise_losses, topk_losses, mse_losses = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        all_y, all_preds = [], []

        for seq_batch, retrain_batch in zip(seq_val_dataloader, retrain_val_dataloader):
            # seq_batch = {key: value.to(self.device) for key, value in seq_batch.items()}
            # cs_batch = {key: value.to(self.device) for key, value in cs_batch.items()}
            seq_x_batch = seq_batch["continuous"]
            true_y = seq_batch["current"]

            # Update weights
            if self.refit:
                refit_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr_refit, weight_decay=0.01)
                retrain_seq_x_batch = retrain_batch["continuous"]
                retrain_true_y = retrain_batch["current"]

                refit_optimizer.zero_grad()
                model.train()
                pred_y = model(retrain_seq_x_batch)

                loss, loss_components = self.val_criterion(pred_y, retrain_true_y)
                icir_loss, list_net, kendall, pairwise, topk, mse = loss_components.values()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                refit_optimizer.step()

            # Predict
            model.eval()
            with torch.inference_mode():
                pred_y = model(seq_x_batch)
                pred_y = torch.nan_to_num(pred_y)

                loss, loss_components = self.val_criterion(pred_y, true_y)
                icir_loss, list_net, kendall, pairwise, topk, mse = loss_components.values()
                losses.append(loss.cpu().numpy())
                icir_losses.append(icir_loss.cpu().numpy())
                list_net_losses.append(list_net.cpu().numpy())
                kendall_losses.append(kendall.cpu().numpy())
                pairwise_losses.append(pairwise.cpu().numpy())
                topk_losses.append(topk.cpu().numpy())
                mse_losses.append(mse.cpu().numpy())

                all_y.append(true_y)
                all_preds.append(pred_y)

        all_y = torch.cat(all_y).detach().cpu().numpy().astype(np.float64)
        all_preds = torch.cat(all_preds).detach().cpu().numpy().astype(np.float64)
        loss = np.mean(losses)
        val_icir_loss = np.mean(icir_losses)
        val_list_net_loss = np.mean(list_net_losses)
        val_kendall_loss = np.mean(kendall_losses)
        val_pairwise_loss = np.mean(pairwise_losses)
        val_topk_loss = np.mean(topk_losses)
        val_mse_loss = np.mean(mse_losses)

        sharpe = self.rank_correlation_sharpe(all_y, all_preds)

        return loss, sharpe, val_icir_loss, val_list_net_loss, val_kendall_loss, val_pairwise_loss, val_topk_loss, val_mse_loss

    def update(self, seq_X: np.array, true_y: np.array):
        """
        Incrementally update the model with new data.

        Parameters
        ----------
        seq_X : np.ndarray
            Input sequence data (seq_len, features).
        true_y : np.ndarray
            Target values for the new sequence.

        Returns
        -------
        None
        """
        torch.manual_seed(self.random_seed)
        if self.lr_refit == 0.0:
            return

        seq_X = torch.tensor(np.nan_to_num(seq_X, nan=0.0), dtype=torch.float32, device=self.device).unsqueeze(0)
        true_y = torch.tensor(np.nan_to_num(true_y, nan=0.0), dtype=torch.float32, device=self.device)
        self.model.train()

        refit_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr_refit, weight_decay=0.01)

        refit_optimizer.zero_grad()
        with torch.autocast(device_type="cuda"):
            pred_y = self.model(seq_X)
            loss, _ = self.update_criterion(pred_y, true_y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        refit_optimizer.step()

    def predict(self, seq_X: np.array) -> np.array:
        """
        Make predictions for a given input sequence.

        Parameters
        ----------
        seq_X : np.ndarray
            Input sequence data (seq_len, features).

        Returns
        -------
        np.ndarray
            Model predictions of shape (1, output_dim).
        """
        torch.manual_seed(self.random_seed)
        seq_X = torch.tensor(np.nan_to_num(seq_X, nan=0.0), dtype=torch.float32, device=self.device).unsqueeze(0)

        self.model.eval()
        with torch.inference_mode():
            preds = self.model(seq_X)
            preds = torch.nan_to_num(preds)

        return preds.cpu().numpy().astype(np.float64)
