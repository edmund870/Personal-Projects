import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Tuple, Dict

from CONFIG import CONFIG
from SEQUENTIAL_NN_MODEL import CNNTransformerModel, GRUModel, LSTMModel, PureTransformerModel, TemporalFusionTransformer
from CROSS_SECTIONAL_NN_MODEL import DeepMLPModel, LinearModel, ResidualMLPModel


class TargetEmbedding(nn.Module):
    """
    Target embedding for asset pairs and lags.
    """

    def __init__(self, target_specs: List[Tuple], pair_emb_dim: int = 32, lag_emb_dim: int = 4):
        super().__init__()

        # Parse target specifications
        self.target_specs = target_specs
        self.num_targets = len(target_specs)

        # Extract unique assets and lags
        unique_pairs = set()
        unique_lags = set()

        for spec in target_specs:
            lag = spec[0]
            pair_str = spec[1]
            unique_lags.add(lag)

            # Keep pairs directional: "A - B" is different from "B - A"
            unique_pairs.add(pair_str.strip())

        # Create mappings
        self.unique_pairs = sorted(list(unique_pairs))
        self.unique_lags = sorted(list(unique_lags))
        self.pair_to_idx = {pair: idx for idx, pair in enumerate(self.unique_pairs)}
        self.lag_to_idx = {lag: idx for idx, lag in enumerate(self.unique_lags)}

        # Embedding layers - one for pairs, one for lags
        self.pair_embedding = nn.Embedding(len(self.unique_pairs), pair_emb_dim)
        self.lag_embedding = nn.Embedding(len(self.unique_lags), lag_emb_dim)

        # Pre-compute target indices
        self.register_buffer("target_indices", self._compute_target_indices())

        self.output_dim = pair_emb_dim + lag_emb_dim

    def _compute_target_indices(self) -> torch.Tensor:
        """Pre-compute indices for all targets"""
        indices = []

        for spec in self.target_specs:
            lag = spec[0]
            pair_str = spec[1]

            pair_idx = self.pair_to_idx[pair_str.strip()]
            lag_idx = self.lag_to_idx[lag]

            indices.append([pair_idx, lag_idx])

        return torch.tensor(indices)

    def forward(self) -> torch.Tensor:
        """
        Returns target embeddings for all 424 targets.

        Returns:
            torch.Tensor: Shape (424, pair_emb_dim + lag_emb_dim)
        """
        # Get embeddings
        pair_emb = self.pair_embedding(self.target_indices[:, 0])  # (424, pair_emb_dim)
        lag_emb = self.lag_embedding(self.target_indices[:, 1])  # (424, lag_emb_dim)

        # Combine embeddings
        target_emb = torch.cat([pair_emb, lag_emb], dim=-1)  # (424, total_emb_dim)

        return target_emb


class ENSEMBLE_NN(nn.Module):
    """
    Enhanced ensemble neural network with target embeddings.

    Combines temporal models with cross-sectional models and target-specific embeddings
    to capture relationships between asset pairs and different lag structures.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, target_specs: List[Tuple], RNN: str = "GRU"):
        super().__init__()

        self.rnn = RNN
        self.output_dim = output_dim  # Should be 424

        # Target embedding
        self.target_embedding = TargetEmbedding(target_specs)
        target_emb_dim = self.target_embedding.output_dim  # 32*2 + 8 = 72

        # Sequential models
        self.gru_model = GRUModel(input_dim, hidden_dim, hidden_dim, num_layers=2)  # Output hidden_dim instead of output_dim

        # Cross-sectional models
        self.mlp = DeepMLPModel(input_dim, hidden_dim, hidden_dim)  # Output hidden_dim instead of output_dim

        # Fusion layers - combine temporal features with target embeddings
        self.temporal_fusion = nn.Sequential(
            nn.Linear(hidden_dim + target_emb_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.cross_sectional_fusion = nn.Sequential(
            nn.Linear(hidden_dim + target_emb_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Final prediction heads
        self.temporal_head = nn.Linear(hidden_dim, 1)
        self.cross_sectional_head = nn.Linear(hidden_dim, 1)

        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(2) / 2)

        # Dropouts
        self.ensemble_dropout = nn.Dropout(0.5)
        self.prediction_dropout = nn.Dropout(0.5)
        self.input_dropout = nn.Dropout(0.5)

    def forward(self, x_seq):
        """
        Forward pass with target embedding integration.

        Parameters:
            x_seq: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: Predictions of shape (batch_size, 424)
        """
        batch_size = x_seq.shape[0]
        x_seq = self.input_dropout(x_seq)

        # Get temporal features
        temporal_features = self.gru_model(x_seq)  # (batch_size, hidden_dim)

        # Get cross-sectional features
        x_cs = x_seq[:, -1, :]  # Last timestep
        cross_sectional_features = self.mlp(x_cs)  # (batch_size, hidden_dim)

        # Get target embeddings (static for all batches)
        target_emb = self.target_embedding()  # (424, target_emb_dim)

        # Expand temporal and cross-sectional features to match targets
        # Broadcast to (batch_size, 424, hidden_dim)
        temporal_expanded = temporal_features.unsqueeze(1).expand(batch_size, self.output_dim, -1)
        cross_sectional_expanded = cross_sectional_features.unsqueeze(1).expand(batch_size, self.output_dim, -1)

        # Expand target embeddings to match batch
        # Broadcast to (batch_size, 424, target_emb_dim)
        target_emb_expanded = target_emb.unsqueeze(0).expand(batch_size, -1, -1)

        # Combine features with target embeddings
        temporal_combined = torch.cat([temporal_expanded, target_emb_expanded], dim=-1)  # (batch_size, 424, hidden_dim + target_emb_dim)
        cross_sectional_combined = torch.cat(
            [cross_sectional_expanded, target_emb_expanded], dim=-1
        )  # (batch_size, 424, hidden_dim + target_emb_dim)

        # Fusion (apply to all targets simultaneously)
        temporal_fused = self.temporal_fusion(temporal_combined)  # (batch_size, 424, hidden_dim)
        cross_sectional_fused = self.cross_sectional_fusion(cross_sectional_combined)  # (batch_size, 424, hidden_dim)

        # Prediction heads
        temporal_pred = self.temporal_head(temporal_fused).squeeze(-1)  # (batch_size, 424)
        cross_sectional_pred = self.cross_sectional_head(cross_sectional_fused).squeeze(-1)  # (batch_size, 424)

        # Apply dropout to predictions
        individual_outputs = [self.prediction_dropout(temporal_pred), self.prediction_dropout(cross_sectional_pred)]

        # Ensemble combination
        dropped_weights = self.ensemble_dropout(self.ensemble_weights)
        weights = F.softmax(dropped_weights, dim=0)

        ensemble_output = torch.zeros_like(individual_outputs[0])  # (batch_size, 424)
        for w, out in zip(weights, individual_outputs):
            ensemble_output += w * out

        # Entropy regularization
        entropy_reg = self.entropy_regularization(weights)
        ensemble_output = ensemble_output + 0.01 * entropy_reg

        # Normalize predictions row-wise (across targets for each batch)
        ensemble_output = (ensemble_output - ensemble_output.mean(dim=1, keepdim=True)) / (ensemble_output.std(dim=1, keepdim=True) + 1e-6)

        return ensemble_output

    def entropy_regularization(self, weights):
        """Compute negative entropy for ensemble weights."""
        return -torch.sum(weights * torch.log(weights + 1e-8))
