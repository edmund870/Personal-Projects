import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional


class DeepMLPModel(nn.Module):
    """MLP with cross-sectional attention for stock relationships"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        # Project raw input to hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Self-attention for cross-sectional relationships
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=0.5, batch_first=True)

        # MLP layers all now operate on hidden_dim
        self.mlp_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, output_dim)])

        # Residual now matches dimensions
        self.residual_proj = nn.Linear(input_dim, output_dim)

        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)])

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x_proj = self.input_projection(x)  # (batch_size, hidden_dim)
        x_expanded = x_proj.unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # Cross-sectional attention
        attn_out, _ = self.cross_attention(x_expanded, x_expanded, x_expanded)
        x_attended = attn_out.squeeze(1)  # (batch_size, hidden_dim)

        # Residual from original input
        residual = self.residual_proj(x)  # (batch_size, output_dim)

        # MLP block
        h = F.relu(self.layer_norms[0](self.mlp_layers[0](x_attended)))
        h = F.relu(self.layer_norms[1](self.mlp_layers[1](h)))
        output = self.mlp_layers[2](h)  # (batch_size, output_dim)

        return output + 0.1 * residual


class LinearModel(nn.Module):
    """
    Linear model for cross-sectional data.

    Applies a single linear transformation to the input features,
    optionally with dropout for regularization.
    """

    def __init__(self, input_dim: int, output_dim: int, use_bias: bool = True):
        """
        Initialize LinearModel.

        Parameters
        ----------
        input_dim : int
            Number of input features.

        output_dim : int
            Number of outputs per sample.

        use_bias : bool, default=True
            Whether to include a bias term in the linear layer.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Linear Model.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, input_dim)
            Input cross-sectional features.

        Returns
        -------
        predictions : torch.Tensor of shape (batch_size, output_dim)
            Predicted values for each sample.
        """
        x_dropped = self.dropout(x)
        predictions = self.linear(x_dropped)
        return predictions


class ResidualMLPModel(nn.Module):
    """
    Residual MLP for cross-sectional data.

    Combines fully-connected layers with residual connections to
    ease optimization for deeper networks.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_blocks: int = 2, dropout: float = 0.2):
        """
        Initialize ResidualMLPModel.

        Parameters
        ----------
        input_dim : int
            Number of input features.

        hidden_dim : int
            Hidden dimension size for residual blocks.

        output_dim : int
            Number of outputs per sample.

        num_blocks : int, default=2
            Number of residual blocks to apply.

        dropout : float, default=0.2
            Dropout probability inside residual blocks.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([self._make_residual_block(hidden_dim, dropout) for _ in range(num_blocks)])
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def _make_residual_block(self, dim: int, dropout: float):
        """
        Create a single residual block.

        Parameters
        ----------
        dim : int
            Input/output dimension of the block.

        dropout : float
            Dropout probability.

        Returns
        -------
        nn.Sequential
            Residual block as a Sequential module.
        """
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Residual MLP.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, input_dim)
            Input cross-sectional features.

        Returns
        -------
        predictions : torch.Tensor of shape (batch_size, output_dim)
            Predicted values for each sample.
        """
        x_hidden = self.input_proj(x)

        for block in self.blocks:
            residual = x_hidden
            x_hidden = block(x_hidden) + residual

        predictions = self.output_head(x_hidden)
        return predictions
