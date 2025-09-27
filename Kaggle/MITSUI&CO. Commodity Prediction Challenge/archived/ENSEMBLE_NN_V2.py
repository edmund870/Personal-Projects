import torch
import torch.nn.functional as F
import torch.nn as nn

from CONFIG import CONFIG
from SEQUENTIAL_NN_MODEL import CNNTransformerModel, GRUModel, LSTMModel, PureTransformerModel
from CROSS_SECTIONAL_NN_MODEL import DeepMLPModel, LinearModel, ResidualMLPModel


class ENSEMBLE_NN(nn.Module):
    """
    Cascading ensemble neural network that predicts features in multiple lags.

    The model predicts 106 features at each lag, concatenates them to the base features,
    and uses the expanded feature set to predict the next lag. This process repeats
    for 4 lags, resulting in a final output of shape (batch_size, 424).

    Parameters
    ----------
    input_dim : int
        Number of base input features per time step.
    hidden_dim : int
        Hidden dimension for sequence models and residual MLP.
    output_dim_per_lag : int, default=106
        Number of features to predict at each lag.
    num_lags : int, default=4
        Number of lag predictions to make.
    RNN : str, default="GRU"
        Type of RNN to use ("GRU" or "LSTM").

    Attributes
    ----------
    output_dim_per_lag : int
        Number of features predicted at each lag.
    num_lags : int
        Total number of lags to predict.
    base_input_dim : int
        Original input dimension (base features).
    ensemble_models : nn.ModuleList
        List of ensemble models for each lag prediction.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim_per_lag: int = 106, num_lags: int = 4, RNN: str = "GRU", compresssion_dim: int = 20
    ):
        super().__init__()

        self.output_dim_per_lag = output_dim_per_lag
        self.num_lags = num_lags
        self.base_input_dim = input_dim
        self.rnn = RNN

        # Create separate ensemble models for each lag
        # Each subsequent model has expanded input dimension
        self.ensemble_models = nn.ModuleList()
        self.compress_layers = nn.ModuleList()
        
        for lag in range(num_lags):
            # Input dimension grows with each lag as we concatenate previous predictions
            current_input_dim = input_dim + (lag * compresssion_dim)
            ensemble = LagEnsembleModel(input_dim=current_input_dim, hidden_dim=hidden_dim, output_dim=output_dim_per_lag, RNN=RNN)
            self.ensemble_models.append(ensemble)

            compress_layer = nn.Linear(output_dim_per_lag, 20)
            self.compress_layers.append(compress_layer)

    def forward(self, x_seq):
        """
        Forward pass with cascading predictions.

        Parameters
        ----------
        x_seq : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Final predictions of shape (batch_size, 424) containing all 4 lag predictions.
        """
        batch_size, seq_len, _ = x_seq.shape

        # Store all lag predictions
        all_predictions = []

        # Current features start with base features
        current_features = x_seq.clone()

        for lag in range(self.num_lags):
            # Predict current lag
            lag_prediction = self.ensemble_models[lag](current_features)
            all_predictions.append(lag_prediction)
            
            compressed_prediction = self.compress_layers[lag](lag_prediction)

            # For next iteration, concatenate prediction to features at each time step
            if lag < self.num_lags - 1:  # Don't expand for the last iteration
                # Expand lag_prediction to match sequence length
                # Shape: (batch_size, 1, output_dim_per_lag) -> (batch_size, seq_len, output_dim_per_lag)
                expanded_prediction = compressed_prediction.unsqueeze(1).expand(-1, seq_len, -1)

                # Concatenate to current features
                current_features = torch.cat([current_features, expanded_prediction], dim=-1)

        # Concatenate all predictions along feature dimension
        final_output = torch.cat(all_predictions, dim=1)  # Shape: (batch_size, 424)

        return final_output


class LagEnsembleModel(nn.Module):
    """
    Single lag ensemble model combining sequence and cross-sectional models.

    This is similar to the original ENSEMBLE_NN but designed for predicting
    a specific number of features for one lag.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, RNN: str = "GRU"):
        super().__init__()

        self.rnn = RNN

        # Sequence models
        if RNN == "GRU":
            self.seq_model = GRUModel(input_dim, hidden_dim, output_dim, num_layers=2)
        elif RNN == "LSTM":
            self.seq_model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers=2)

        # Cross-sectional model
        self.mlp = DeepMLPModel(input_dim, hidden_dim, output_dim)

        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(2) / 2)

        # Dropout layers
        self.ensemble_dropout = nn.Dropout(0.5)
        self.prediction_dropout = nn.Dropout(0.5)
        self.input_dropout = nn.Dropout(0.5)

    def forward(self, x_seq):
        """
        Forward pass for single lag prediction.

        Parameters
        ----------
        x_seq : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Predictions of shape (batch_size, output_dim).
        """
        x_seq = self.input_dropout(x_seq)

        # Get predictions from sequence and cross-sectional models
        x_cs = x_seq[:, -1, :]  # Use last time step for cross-sectional model

        seq_out = self.seq_model(x_seq)
        mlp_out = self.mlp(x_cs)

        individual_outputs = [seq_out, mlp_out]
        individual_outputs = [self.prediction_dropout(out) for out in individual_outputs]

        # Apply dropout to ensemble weights and normalize
        dropped_weights = self.ensemble_dropout(self.ensemble_weights)
        weights = F.softmax(dropped_weights, dim=0)

        # Combine predictions
        ensemble_output = torch.zeros_like(individual_outputs[0])
        for w, out in zip(weights, individual_outputs):
            ensemble_output += w * out

        # Add entropy regularization
        entropy_reg = self.entropy_regularization(weights)
        ensemble_output = ensemble_output + 0.01 * entropy_reg

        # Normalize output
        ensemble_output = (ensemble_output - ensemble_output.mean(dim=1, keepdim=True)) / (ensemble_output.std(dim=1, keepdim=True) + 1e-8)

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
            Negative entropy (-sum w_i * log(w_i)).
        """
        return -torch.sum(weights * torch.log(weights + 1e-8))
