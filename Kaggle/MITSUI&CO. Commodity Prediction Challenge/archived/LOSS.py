import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from typing import Tuple, Dict, Any, Optional


class SpearmanCorrelationLoss(nn.Module):
    """
    Compute Spearman correlation for a single batch.

    This is used as a building block for ICIR-based losses.
    Computes Spearman correlation for each sample in a batch by ranking
    predictions and true values and computing Pearson correlation of the ranks.
    """

    def __init__(self, eps: float = 1e-8):
        """
        Initialize SpearmanCorrelationLoss.

        Parameters
        ----------
        eps : float, default=1e-8
            Small constant to avoid division by zero.
        """
        super().__init__()
        self.eps = eps

    def forward(self, pred_y: torch.Tensor, true_y: torch.Tensor) -> torch.Tensor:
        """
        Compute Spearman correlation for each sample in the batch.

        Parameters
        ----------
        pred_y : torch.Tensor of shape (batch_size, n_assets)
            Predicted values.

        true_y : torch.Tensor of shape (batch_size, n_assets)
            True values.

        Returns
        -------
        correlations : torch.Tensor of shape (batch_size,)
            Spearman correlation for each sample.
        """
        batch_size = pred_y.shape[0]
        correlations = []

        for i in range(batch_size):
            valid_mask = ~torch.isnan(true_y[i])
            valid_pred = pred_y[i][valid_mask]
            valid_true = true_y[i][valid_mask]

            pred_ranks = self._get_average_ranks(valid_pred)
            true_ranks = self._get_average_ranks(valid_true)

            correlation = self._pearson_correlation(pred_ranks, true_ranks)
            correlations.append(correlation)

        return torch.stack(correlations)

    def _get_average_ranks(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert values to average ranks (handling ties).

        Parameters
        ----------
        x : torch.Tensor
            Input values.

        Returns
        -------
        ranks : torch.Tensor
            Average ranks corresponding to input values.
        """
        sorted_vals, sorted_indices = torch.sort(x)
        ranks = torch.empty_like(x, dtype=torch.float)
        unique_vals, inverse_indices = torch.unique(sorted_vals, return_inverse=True)
        counts = torch.bincount(inverse_indices)
        cumsum = torch.cumsum(counts, dim=0)
        group_starts = torch.zeros_like(cumsum)
        group_starts[1:] = cumsum[:-1]
        avg_ranks = (group_starts + cumsum - 1) / 2.0 + 1
        group_ranks = avg_ranks[inverse_indices]
        ranks[sorted_indices] = group_ranks
        return ranks

    def _pearson_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute Pearson correlation between two tensors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        y : torch.Tensor
            Input tensor.

        Returns
        -------
        corr : torch.Tensor
            Pearson correlation scalar.
        """
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        numerator = (x_centered * y_centered).sum()
        denominator = torch.sqrt((x_centered**2).sum() * (y_centered**2).sum() + self.eps)
        return numerator / denominator


class RankICIRLoss(nn.Module):
    """
    Rank Information Coefficient Information Ratio (ICIR) loss.

    Computes the ICIR = mean(Spearman_correlations) / std(Spearman_correlations)
    over a memory window of correlations. Encourages high mean correlation
    and low volatility in correlations.
    """

    def __init__(
        self,
        memory_size: int = 90,
        stability_weight: float = 0.1,
        min_samples: int = 30,
        eps: float = 1e-8,
    ):
        """
        Initialize RankICIRLoss.

        Parameters
        ----------
        memory_size : int, default=1000
            Number of recent correlations to keep in memory.

        stability_weight : float, default=0.1
            Weight for stability term (reduces volatility of correlations).

        min_samples : int, default=90
            Minimum number of samples required before computing ICIR.

        eps : float, default=1e-8
            Small constant for numerical stability.
        """
        super().__init__()
        self.memory_size = memory_size
        self.stability_weight = stability_weight
        self.min_samples = min_samples
        self.eps = eps

        self.register_buffer("correlation_memory", torch.zeros(memory_size))
        self.register_buffer("memory_pointer", torch.tensor(0, dtype=torch.long))
        self.register_buffer("memory_filled", torch.tensor(False))

        self.spearman_loss = SpearmanCorrelationLoss(eps)

    def forward(self, pred_y: torch.Tensor, true_y: torch.Tensor) -> torch.Tensor:
        """
        Compute Rank ICIR loss for a batch.

        Parameters
        ----------
        pred_y : torch.Tensor of shape (batch_size, n_assets)
            Predictions.

        true_y : torch.Tensor of shape (batch_size, n_assets)
            True values.

        Returns
        -------
        loss : torch.Tensor
            Scalar ICIR loss (negative ICIR with stability penalty).
        """
        batch_correlations = self.spearman_loss(pred_y, true_y)
        self._update_memory(batch_correlations)

        if self.memory_filled:
            correlations = self.correlation_memory
        else:
            valid_size = min(self.memory_pointer.item(), self.memory_size)
            if valid_size < self.min_samples:
                return -batch_correlations.mean()
            correlations = self.correlation_memory[:valid_size]

        mean_corr = correlations.mean()
        std_corr = torch.sqrt(((correlations - mean_corr) ** 2).mean()) + self.eps
        icir = mean_corr / std_corr
        current_batch_std = batch_correlations.std(unbiased=False) + self.eps
        stability_penalty = self.stability_weight * current_batch_std

        loss = -icir + stability_penalty
        return loss

    def _update_memory(self, correlations: torch.Tensor):
        """
        Update circular buffer with new batch correlations.

        Parameters
        ----------
        correlations : torch.Tensor
            Spearman correlations for current batch.
        """
        batch_size = correlations.shape[0]
        for i in range(batch_size):
            idx = self.memory_pointer % self.memory_size
            self.correlation_memory[idx] = correlations[i]
            self.memory_pointer += 1
            if self.memory_pointer >= self.memory_size:
                self.memory_filled = torch.tensor(True)

    def get_current_icir(self) -> float:
        """
        Get current ICIR value from memory.

        Returns
        -------
        float
            Current ICIR value.
        """
        if self.memory_filled:
            correlations = self.correlation_memory
        else:
            valid_size = min(self.memory_pointer.item(), self.memory_size)
            if valid_size < self.min_samples:
                return 0.0
            correlations = self.correlation_memory[:valid_size]

        mean_corr = correlations.mean().item()
        std_corr = torch.sqrt(((correlations - mean_corr) ** 2).mean()).item() + self.eps
        return mean_corr / std_corr

    def get_correlation_stats(self) -> Dict[str, float]:
        """
        Get detailed correlation statistics from memory.

        Returns
        -------
        stats : dict
            Dictionary containing mean, std, min, max, ICIR, and sample count.
        """
        if self.memory_filled:
            correlations = self.correlation_memory
        else:
            valid_size = min(self.memory_pointer.item(), self.memory_size)
            if valid_size == 0:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "icir": 0.0, "samples": 0}
            correlations = self.correlation_memory[:valid_size]

        return {
            "mean": correlations.mean().item(),
            "std": correlations.std(unbiased=True).item(),
            "min": correlations.min().item(),
            "max": correlations.max().item(),
            "icir": self.get_current_icir(),
            "samples": len(correlations),
        }


class CombinedICIRLoss(nn.Module):
    """
    Combined loss for ICIR, MSE, and ranking objectives.

    The total loss is:
        total_loss = icir_weight * ICIR_loss
                   + mse_weight * MSE_loss
                   + ranking_weight * ranking_loss
    """

    def __init__(self, icir_weight: float = 1.0, mse_weight: float = 20.0, huber_weight: float = 1.0, ranking_weight: float = 1.0, **icir_kwargs):
        """
        Initialize CombinedICIRLoss.

        Parameters
        ----------
        icir_weight : float, default=2.0
            Weight of ICIR loss.

        mse_weight : float, default=10.0
            Weight of MSE loss.

        huber_weight : float, default=50.0
            Weight of ranking loss.

        icir_kwargs : dict
            Additional keyword arguments to pass to RankICIRLoss.
        """
        super().__init__()
        self.icir_weight = icir_weight
        self.mse_weight = mse_weight
        self.huber_weight = huber_weight
        self.ranking_weight = ranking_weight

        self.icir_loss = RankICIRLoss(**icir_kwargs)
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(delta=1.0)

    def forward(self, pred_y: torch.Tensor, true_y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Parameters
        ----------
        pred_y : torch.Tensor of shape (batch_size, n_assets)
            Predictions.

        true_y : torch.Tensor of shape (batch_size, n_assets)
            True values.

        Returns
        -------
        losses : dict
            Dictionary containing:
            - total_loss : scalar tensor
            - icir_loss : scalar tensor
            - mse_loss : scalar tensor
            - huber_loss : scalar tensor
            - current_icir : scalar tensor
        """
        icir_loss = self.icir_loss(pred_y, true_y)
        valid_mask = ~torch.isnan(true_y)
        mse_loss = self.mse_loss(pred_y[valid_mask], true_y[valid_mask])
        huber_loss = self.huber_loss(pred_y[valid_mask], true_y[valid_mask])
        ranking_loss = self._compute_ranking_loss(pred_y, true_y)

        total_loss = self.icir_weight * icir_loss + self.mse_weight * mse_loss + self.huber_weight * huber_loss + self.ranking_weight * ranking_loss

        return {
            "total_loss": total_loss,
            "icir_loss": icir_loss,
            "mse_loss": mse_loss,
            "huber_loss": huber_loss,
            "ranking_loss": ranking_loss,
            "current_icir": torch.tensor(self.icir_loss.get_current_icir()),
        }

    def _compute_ranking_loss(self, pred_y: torch.Tensor, true_y: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise ranking loss.

        This encourages the predicted relative ordering of assets to match
        the true ordering.

        Parameters
        ----------
        pred_y : torch.Tensor of shape (batch_size, n_assets)
            Predictions.

        true_y : torch.Tensor of shape (batch_size, n_assets)
            True values.

        Returns
        -------
        ranking_loss : torch.Tensor
            Scalar tensor representing the average pairwise ranking disagreement.
        """
        batch_size, n_assets = pred_y.shape
        total_loss = 0.0

        for i in range(batch_size):
            valid_mask = ~torch.isnan(true_y[i])
            pred_i = pred_y[i][valid_mask]
            true_i = true_y[i][valid_mask]

            pred_diff = pred_i.unsqueeze(1) - pred_i.unsqueeze(0)
            true_diff = true_i.unsqueeze(1) - true_i.unsqueeze(0)

            sign_pred = torch.sign(pred_diff)
            sign_true = torch.sign(true_diff)

            disagreement = (sign_pred != sign_true).float()
            weights = torch.abs(true_diff) + 1e-8
            weighted_disagreement = disagreement * weights

            total_loss += weighted_disagreement.mean()

        return total_loss / batch_size
