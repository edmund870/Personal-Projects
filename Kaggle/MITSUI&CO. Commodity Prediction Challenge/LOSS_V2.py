import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from typing import Tuple, Dict, Any, Optional


def create_valid_mask(targets):
    """Create mask for non-NaN values"""
    return ~torch.isnan(targets)


def apply_nan_mask(predictions, targets, mask):
    """Apply NaN mask to both predictions and targets"""
    # For each sample in batch, only keep valid (non-NaN) elements
    valid_preds = []
    valid_targets = []

    for i in range(predictions.shape[0]):
        sample_mask = mask[i]
        if sample_mask.sum() > 1:  # Need at least 2 valid values for correlation
            valid_preds.append(predictions[i][sample_mask])
            valid_targets.append(targets[i][sample_mask])
        else:
            # If less than 2 valid values, create dummy tensors (will be masked out later)
            valid_preds.append(torch.zeros(2, device=predictions.device))
            valid_targets.append(torch.zeros(2, device=predictions.device))

    return valid_preds, valid_targets


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


class ListNetLoss(nn.Module):
    """ListNet loss for learning to rank with NaN handling"""

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, predictions, targets):
        # Create mask for valid values
        valid_mask = create_valid_mask(targets)
        valid_samples = valid_mask.sum(dim=-1) > 1

        if not valid_samples.any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        batch_losses = []

        for i in range(predictions.shape[0]):
            if not valid_samples[i]:
                continue

            sample_mask = valid_mask[i]
            pred_sample = predictions[i][sample_mask]
            target_sample = targets[i][sample_mask]

            # Convert to probability distributions
            pred_probs = F.softmax(pred_sample / self.temperature, dim=-1)
            target_probs = F.softmax(target_sample / self.temperature, dim=-1)

            # KL divergence for this sample
            sample_loss = F.kl_div(pred_probs.log(), target_probs, reduction="sum")
            batch_losses.append(sample_loss)

        if not batch_losses:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        return torch.stack(batch_losses).mean()


class KendallTauLoss(nn.Module):
    """Differentiable approximation of Kendall's Tau with NaN handling"""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, predictions, targets):
        # Create mask for valid values
        valid_mask = create_valid_mask(targets)
        valid_samples = valid_mask.sum(dim=-1) > 1

        if not valid_samples.any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        batch_concordances = []

        for i in range(predictions.shape[0]):
            if not valid_samples[i]:
                continue

            sample_mask = valid_mask[i]
            pred_sample = predictions[i][sample_mask]
            target_sample = targets[i][sample_mask]

            n = len(pred_sample)
            if n < 2:
                continue

            # Create all pairwise combinations
            indices = torch.combinations(torch.arange(n, device=predictions.device), 2)
            i_indices = indices[:, 0]
            j_indices = indices[:, 1]

            # Get pairs
            pred_i = pred_sample[i_indices]
            pred_j = pred_sample[j_indices]
            target_i = target_sample[i_indices]
            target_j = target_sample[j_indices]

            # Compute concordance (soft)
            pred_diff = pred_i - pred_j
            target_diff = target_i - target_j

            # Soft sign function using tanh
            pred_sign = torch.tanh(pred_diff / self.temperature)
            target_sign = torch.tanh(target_diff / self.temperature)

            # Concordance score
            concordance = (pred_sign * target_sign).mean()
            batch_concordances.append(concordance)

        if not batch_concordances:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # Return negative concordance (minimize to maximize)
        return -torch.stack(batch_concordances).mean()


class PairwiseRankingLoss(nn.Module):
    """Pairwise ranking loss with margin and NaN handling"""

    def __init__(self, margin=1.0, temperature=0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, predictions, targets):
        # Create mask for valid values
        valid_mask = create_valid_mask(targets)
        valid_samples = valid_mask.sum(dim=-1) > 1

        if not valid_samples.any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        batch_losses = []

        for i in range(predictions.shape[0]):
            if not valid_samples[i]:
                continue

            sample_mask = valid_mask[i]
            pred_sample = predictions[i][sample_mask]
            target_sample = targets[i][sample_mask]

            n = len(pred_sample)
            if n < 2:
                continue

            # Create all pairs for this sample
            pred_i = pred_sample.unsqueeze(1).expand(-1, n)  # (n, n)
            pred_j = pred_sample.unsqueeze(0).expand(n, -1)  # (n, n)
            target_i = target_sample.unsqueeze(1).expand(-1, n)
            target_j = target_sample.unsqueeze(0).expand(n, -1)

            # Only consider pairs where i != j
            mask = torch.eye(n, device=predictions.device) == 0

            # Target ordering: should pred_i > pred_j when target_i > target_j?
            target_order = (target_i > target_j).float()

            # Prediction difference
            pred_diff = pred_i - pred_j

            # Ranking loss: when target_i > target_j, we want pred_i - pred_j > margin
            loss = torch.relu(self.margin - pred_diff) * target_order
            loss = loss + torch.relu(self.margin + pred_diff) * (1 - target_order)

            # Apply mask and average
            sample_loss = (loss * mask.float()).sum() / mask.sum().float()
            batch_losses.append(sample_loss)

        if not batch_losses:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        return torch.stack(batch_losses).mean()


class TopKRankingLoss(nn.Module):
    """Focus on correctly ranking top and bottom K elements with NaN handling"""

    def __init__(self, k_percent=0.2, temperature=0.1):
        super().__init__()
        self.k_percent = k_percent
        self.temperature = temperature

    def forward(self, predictions, targets):
        # Create mask for valid values
        valid_mask = create_valid_mask(targets)
        valid_samples = valid_mask.sum(dim=-1) > 1

        if not valid_samples.any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        batch_losses = []

        for i in range(predictions.shape[0]):
            if not valid_samples[i]:
                continue

            sample_mask = valid_mask[i]
            pred_sample = predictions[i][sample_mask]
            target_sample = targets[i][sample_mask]

            n = len(pred_sample)
            k = max(1, int(n * self.k_percent))

            if k >= n // 2:  # Need enough elements for top and bottom k
                continue

            # Get top-k and bottom-k indices for targets
            _, top_k_indices = torch.topk(target_sample, k, dim=-1)
            _, bottom_k_indices = torch.topk(target_sample, k, dim=-1, largest=False)

            # Get corresponding predictions
            top_k_preds = pred_sample[top_k_indices]
            bottom_k_preds = pred_sample[bottom_k_indices]

            # Loss: top-k predictions should be higher than bottom-k predictions
            top_mean = top_k_preds.mean()
            bottom_mean = bottom_k_preds.mean()

            # We want top_mean > bottom_mean
            margin_loss = torch.relu(1.0 - (top_mean - bottom_mean))
            batch_losses.append(margin_loss)

        if not batch_losses:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        return torch.stack(batch_losses).mean()


class CombinedICIRLoss(nn.Module):
    """
    Combined loss for ICIR, MSE, and ranking objectives.

    The total loss is:
        total_loss = icir_weight * ICIR_loss
                   + mse_weight * MSE_loss
                   + ranking_weight * ranking_loss
    """

    """
    Ensemble of ranking losses optimized for maximizing Spearman correlation
    and ultimately improving ICIR (mean/std of correlations across time)
    """

    def __init__(
        self,
        spearman_weight=0.4,
        listnet_weight=0.2,
        kendall_weight=0.15,
        pairwise_weight=0.1,
        topk_weight=0.1,
        quantile_weight=0.05,
        mse_weight=0.1,  # Small MSE for stability
        listnet_temp=1.0,
        kendall_temp=0.1,
        **icir_kwargs,
    ):
        super().__init__()

        # Loss components
        self.spearman_loss = RankICIRLoss(**icir_kwargs)
        self.listnet_loss = ListNetLoss(temperature=listnet_temp)
        self.kendall_loss = KendallTauLoss(temperature=kendall_temp)
        self.pairwise_loss = PairwiseRankingLoss(margin=1.0, temperature=0.1)
        self.topk_loss = TopKRankingLoss(k_percent=0.1)

        self.mse_loss = nn.MSELoss()

        # Weights
        self.weights = {
            "spearman": spearman_weight,
            "listnet": listnet_weight,
            "kendall": kendall_weight,
            "pairwise": pairwise_weight,
            "topk": topk_weight,
            "quantile": quantile_weight,
            "mse": mse_weight,
        }

        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def forward(self, predictions, targets, return_components=False):
        """
        Args:
            predictions: (batch_size, 424) model predictions
            targets: (batch_size, 424) true targets (Gaussian ranked, may contain NaNs)
            return_components: if True, return dict with individual losses
        """

        # Check if we have any valid data
        valid_mask = create_valid_mask(targets)
        has_valid_data = (valid_mask.sum(dim=-1) > 1).any()

        if not has_valid_data:
            # Return zero loss if no valid data
            dummy_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
            if return_components:
                losses = {k: dummy_loss for k in self.weights.keys()}
                return dummy_loss, losses
            return dummy_loss

        losses = {}

        # Core ranking losses
        losses["spearman"] = self.spearman_loss(predictions, targets)
        losses["listnet"] = self.listnet_loss(predictions, targets)
        losses["kendall"] = self.kendall_loss(predictions, targets)

        # Pairwise and structural losses
        losses["pairwise"] = self.pairwise_loss(predictions, targets)
        losses["topk"] = self.topk_loss(predictions, targets)
        # losses["quantile"] = self.quantile_loss(predictions, targets)

        # Stability loss (MSE with NaN masking)
        losses["mse"] = self.mse_loss(predictions[valid_mask], targets[valid_mask])

        # Weighted combination
        total_loss = sum(self.weights[k] * losses[k] for k in losses.keys())

        return total_loss, losses
