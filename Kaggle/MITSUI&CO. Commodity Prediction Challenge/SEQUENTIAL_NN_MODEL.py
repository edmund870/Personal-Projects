import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional


class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encoding to input embeddings for temporal awareness.

    Parameters
    ----------
    d_model : int
        Dimensionality of the input embeddings.
    max_len : int, optional
        Maximum length of sequences (default 5000).

    Methods
    -------
    forward(x)
        Adds positional encodings to the input tensor.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (seq_len, batch_size, d_model).

        Returns
        -------
        torch.Tensor
            Input tensor with positional encodings added.
        """
        return x + self.pe[: x.size(0), :]


class CNN1DExtractor(nn.Module):
    """
    1D CNN for local pattern extraction across sequences.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dim : int
        Number of hidden channels in CNN layers.
    num_layers : int, optional
        Number of convolutional layers (default 3).

    Methods
    -------
    forward(x)
        Apply 1D convolutions with residual connections and return transformed sequence.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_channels = input_dim
        for i in range(num_layers):
            kernel_size = 3 + 2 * i
            self.convs.append(nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=kernel_size // 2))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            in_channels = hidden_dim
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 1D convolutions with residual connections.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, hidden_dim).
        """
        x = x.transpose(1, 2)
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = F.relu(norm(conv(x)))
            if x.shape == residual.shape:
                x = x + residual
            x = self.dropout(x)
        return x.transpose(1, 2)


class TransformerEncoder(nn.Module):
    """
    Standard Transformer Encoder for temporal modeling.

    Parameters
    ----------
    d_model : int
        Dimension of input embeddings.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of Transformer encoder layers.
    dim_feedforward : int
        Dimension of feedforward layers.

    Methods
    -------
    forward(x)
        Encode input sequence using Transformer.
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence using Transformer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model).

        Returns
        -------
        torch.Tensor
            Encoded tensor of shape (batch_size, seq_len, d_model).
        """
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        return self.transformer_encoder(x)


class CrossSectionalAttention(nn.Module):
    """
    Cross-asset attention for capturing relationships between features/assets.

    Parameters
    ----------
    d_model : int
        Dimension of input features.
    nhead : int
        Number of attention heads.

    Methods
    -------
    forward(x)
        Apply multi-head attention and feedforward layers.
    """

    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-sectional attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        attn_out, _ = self.multihead_attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class CNNTransformerModel(nn.Module):
    """
    Hybrid CNN + Transformer model for cross-sectional time series prediction.

    Combines local pattern extraction (CNN), temporal modeling (Transformer),
    and cross-sectional attention for asset relationships.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step.
    hidden_dim : int
        Dimension of hidden layers.
    output_dim : int
        Number of outputs per sample.
    seq_len : int
        Sequence length of input time series.
    num_heads : int, optional
        Number of attention heads (default 4).
    num_layers : int, optional
        Number of Transformer layers (default 2).

    Methods
    -------
    forward(x)
        Forward pass to predict output for the last time step.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seq_len: int, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.cnn_extractor = CNN1DExtractor(hidden_dim, hidden_dim)
        self.transformer = TransformerEncoder(hidden_dim, num_heads, num_layers, hidden_dim * 4)
        self.cross_attention = CrossSectionalAttention(hidden_dim, num_heads)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        self.skip_projection = nn.Linear(input_dim, output_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CNN-Transformer model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted tensor of shape (batch_size, output_dim).
        """
        skip_out = self.skip_projection(x[:, -1, :])
        x = self.input_projection(x)
        x = self.cnn_extractor(x)
        x = self.transformer(x)
        x = self.cross_attention(x)
        x_last = self.final_norm(x[:, -1, :])
        head_out = self.head(x_last)
        return head_out + 0.1 * skip_out


class GRUModel(nn.Module):
    """
    Bidirectional GRU model with self-attention for sequence prediction.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step.
    hidden_dim : int
        Number of hidden units in GRU.
    output_dim : int
        Number of outputs per sample.
    num_layers : int, optional
        Number of GRU layers (default 2).

    Methods
    -------
    forward(x)
        Forward pass to produce output at last time step.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, 8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GRU model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted tensor of shape (batch_size, output_dim).
        """
        self.gru.flatten_parameters()
        gru_out, _ = self.gru(x)
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        gru_out = self.norm(gru_out + attn_out)
        return self.output_head(gru_out[:, -1, :])


class LSTMModel(nn.Module):
    """
    Bidirectional LSTM model with self-attention for sequence prediction.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step.
    hidden_dim : int
        Number of hidden units in LSTM.
    output_dim : int
        Number of outputs per sample.
    num_layers : int, optional
        Number of LSTM layers (default 2).

    Methods
    -------
    forward(x)
        Forward pass to produce output at last time step.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, 8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of LSTM model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted tensor of shape (batch_size, output_dim).
        """
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        lstm_out = self.norm(lstm_out + attn_out)
        return self.output_head(lstm_out[:, -1, :])


class PureTransformerModel(nn.Module):
    """
    Transformer-only model for sequence prediction.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step.
    hidden_dim : int
        Dimension of hidden layers.
    output_dim : int
        Number of outputs per sample.
    num_heads : int, optional
        Number of attention heads (default 4).
    num_layers : int, optional
        Number of Transformer layers (default 2).

    Methods
    -------
    forward(x)
        Forward pass to predict output at last time step.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.transformer = TransformerEncoder(hidden_dim, num_heads, num_layers, hidden_dim * 4)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Pure Transformer model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted tensor of shape (batch_size, output_dim).
        """
        x = self.input_projection(x)
        x = self.transformer(x)
        return self.output_head(x[:, -1, :])


class GateAddNorm(nn.Module):
    """Gated residual connection + layer norm (TFT §3.2)."""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(d_model))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, residual):
        return self.norm(residual + self.dropout(x * self.gate))


class VariableSelectionNetwork(nn.Module):
    """Learns per-feature weights for variable importance."""

    def __init__(self, n_vars, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.context_fc = nn.Linear(d_model, n_vars)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_emb):
        # Use first timestep mean as context (B, n_vars, d_model) -> (B, n_vars)
        context = x_emb[:, 0].mean(dim=1)  # (B, n_vars)
        weights = self.softmax(self.context_fc(context))  # (B, n_vars)
        # Apply weights: (B, L, n_vars, d_model) * (B, 1, n_vars, 1) -> sum over n_vars
        v = (x_emb * weights.unsqueeze(1).unsqueeze(-1)).sum(dim=2)  # (B, L, d_model)
        return self.dropout(v), weights


class InterpretableMultiHeadAttention(nn.Module):
    """Standard MHA with interpretability."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.mha(x, x, x, key_padding_mask=mask)
        return self.dropout(attn_out)


class TFTBlock(nn.Module):
    """Single TFT transformer block with gated residuals."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attn = InterpretableMultiHeadAttention(d_model, n_heads, dropout)
        self.gate_norm = GateAddNorm(d_model, dropout)

        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))
        self.gate_norm_ff = GateAddNorm(d_model, dropout)

    def forward(self, x, mask=None):
        x = self.gate_norm(self.attn(x, mask), x)
        return self.gate_norm_ff(self.ff(x), x)


class TemporalFusionTransformer(nn.Module):
    """Minimal TFT for continuous inputs and multi-target regression."""

    def __init__(self, n_vars: int, n_targets: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        # Feature embedding
        self.feature_embed = nn.Linear(1, d_model)

        # Variable selection network
        self.vsn = VariableSelectionNetwork(n_vars, d_model, dropout)

        # TFT blocks
        self.blocks = nn.ModuleList([TFTBlock(d_model, n_heads, dropout) for _ in range(n_layers)])

        # Output head
        self.head = nn.Linear(d_model, n_targets)

    def forward(self, x, mask=None):
        """
        x: (B, L, n_vars) - your sequential input
        returns: predictions (B, L, n_targets), variable_weights (B, n_vars)
        """
        # Embed each variable separately
        x_emb = self.feature_embed(x.unsqueeze(-1))  # (B, L, n_vars, d_model)

        # Variable selection
        x, var_weights = self.vsn(x_emb)  # (B, L, d_model)

        # Pass through TFT blocks
        for block in self.blocks:
            x = block(x, mask)

        predictions = self.head(x)  # (B, L, n_targets)
        return predictions[:, -1, :], var_weights
