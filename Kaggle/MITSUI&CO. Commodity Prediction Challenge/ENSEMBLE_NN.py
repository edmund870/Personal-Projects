import torch
import torch.nn.functional as F
import torch.nn as nn

from CONFIG import CONFIG
from SEQUENTIAL_NN_MODEL import CNNTransformerModel, GRUModel, LSTMModel, PureTransformerModel, TemporalFusionTransformer
from CROSS_SECTIONAL_NN_MODEL import DeepMLPModel, LinearModel, ResidualMLPModel


class ENSEMBLE_NN(nn.Module):
    """
    Ensemble neural network combining multiple sequence and cross-sectional models.

    The ensemble includes sequence models (GRU, CNN-Transformer, LSTM, Pure Transformer)
    and cross-sectional models (Deep MLP, Linear, Residual MLP). Predictions are combined
    using learnable weights, with entropy regularization to prevent overconfidence.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step (for sequence models) or per asset (for cross-sectional models).
    hidden_dim : int
        Hidden dimension for sequence models and residual MLP.
    output_dim : int
        Number of output predictions (e.g., number of assets).

    Attributes
    ----------
    cnn_transformer : nn.Module
        CNN + Transformer hybrid model for temporal features.
    gru_model : nn.Module
        GRU-based model for temporal sequences.
    lstm_model : nn.Module
        LSTM-based model for temporal sequences.
    pure_transformer : nn.Module
        Pure transformer for temporal sequences.
    mlp : nn.Module
        Deep MLP for cross-sectional features.
    linear : nn.Module
        Linear model for cross-sectional features.
    residual : nn.Module
        Residual MLP for cross-sectional features.
    ensemble_weights : nn.Parameter
        Learnable weights for combining individual model predictions.
    input_dropout : nn.Dropout
        Dropout applied to input features.
    prediction_dropout : nn.Dropout
        Dropout applied to individual model predictions.
    ensemble_dropout : nn.Dropout
        Dropout applied to ensemble weights.

    Methods
    -------
    forward(x_seq)
        Compute ensemble predictions from input sequence.
    entropy_regularization(weights)
        Compute negative entropy of ensemble weights to encourage diversity.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, RNN: str = "GRU"):
        super().__init__()

        self.rnn = RNN
        # seq models
        # self.cnn_transformer = CNNTransformerModel(input_dim, hidden_dim, output_dim, CONFIG.SEQ_LEN)
        self.gru_model = GRUModel(input_dim, hidden_dim, output_dim, num_layers=2)
        # self.lstm_model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers=2)
        # self.pure_transformer = PureTransformerModel(input_dim, hidden_dim, output_dim)
        # self.tft_model = TemporalFusionTransformer(
        #     n_vars=input_dim,
        #     n_targets=output_dim,
        #     d_model=hidden_dim,
        #     n_heads=4,  # or 4 if hidden_dim not divisible by 8
        #     n_layers=2,
        #     dropout=0.4,
        # )

        # cross sectional models
        self.mlp = DeepMLPModel(input_dim, hidden_dim, output_dim)
        # self.linear = LinearModel(input_dim, output_dim)
        # self.residual = ResidualMLPModel(input_dim, hidden_dim, output_dim)

        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(2) / 2)

        self.ensemble_dropout = nn.Dropout(0.5)
        self.prediction_dropout = nn.Dropout(0.5)
        self.input_dropout = nn.Dropout(0.5)

    def forward(
        self,
        x_seq,
    ):
        """
        Compute ensemble predictions.

        Combines predictions from sequence models (GRU here) and cross-sectional models (MLP here)
        using learnable softmax-normalized weights, with entropy regularization.

        Parameters
        ----------
        x_seq : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim), representing
            sequences for multiple assets.

        Returns
        -------
        torch.Tensor
            Ensemble predictions of shape (batch_size, output_dim), with entropy
            regularization applied.

        Notes
        -----
        Currently, only GRU and MLP models are active in the ensemble.
        Other models (CNN-Transformer, LSTM, Pure Transformer, Linear, Residual) are defined
        but commented out for future activation.
        """
        x_seq = self.input_dropout(x_seq)
        # Get predictions from all models
        x_cs = x_seq[:, -1, :]
        # out1 = self.cnn_transformer(x_seq)
        if self.rnn == "GRU":
            out2 = self.gru_model(x_seq)
        # out2, _ = self.tft_model(x_seq)
        # if self.rnn == "LSTM":
        #     out2 = self.lstm_model(x_seq)
        # out4 = self.pure_transformer(x_seq)
        out5 = self.mlp(x_cs)
        # out6 = self.linear(x_cs)
        # out7 = self.residual(x_cs)

        individual_outputs = [out2, out5]  # out1, out2, out3, out4, out5, out6, out7
        individual_outputs = [self.prediction_dropout(out) for out in individual_outputs]

        dropped_weights = self.ensemble_dropout(self.ensemble_weights)
        weights = F.softmax(dropped_weights, dim=0)

        ensemble_output = torch.zeros_like(individual_outputs[0])  # (batch_size, 424)
        for w, out in zip(weights, individual_outputs):
            ensemble_output += w * out

        entropy_reg = self.entropy_regularization(weights)

        ensemble_output = ensemble_output + 0.01 * entropy_reg

        ensemble_output = (ensemble_output - ensemble_output.mean(dim=1, keepdim=True)) / (ensemble_output.std(dim=1, keepdim=True) + 1e-6)

        return ensemble_output

    def entropy_regularization(self, weights):
        """
        Compute negative entropy for ensemble weights.

        Parameters
        ----------
        weights : torch.Tensor
            Softmax-normalized ensemble weights.

        Returns
        -------
        torch.Tensor
            Negative entropy (-sum w_i * log(w_i)), used to penalize overconfident weights.
        """
        return -torch.sum(weights * torch.log(weights + 1e-8))  # Entropy term
