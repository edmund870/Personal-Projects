from typing import Dict, List, Optional
from abc import ABC, abstractmethod

import polars as pl
import torch
from torch.utils.data import Dataset

from CONFIG import CONFIG


class BaseFinancialDataset(Dataset, ABC):
    """Base class for financial datasets"""

    def __init__(self, X: pl.DataFrame, y: pl.DataFrame, date_column: str = CONFIG.DATE_COL):
        """
        Base initialization

        Args:
            data: Preprocessed DataFrame (scaling already done)
            target_columns: Target column names (allocation)
            feature_columns: List of feature column names
            date_column: Name of date identifier column
        """
        self.X = X.clone()
        self.y = y.clone()
        self.date_column = date_column

        # Sort by date
        self.X = self.X.sort(by=CONFIG.DATE_COL)
        self.y = self.y.sort(by=CONFIG.DATE_COL)
        self.unique_dates = sorted(self.X[self.date_column].unique())
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._prepare_samples()

    def _prepare_samples(self):
        self.dates = torch.tensor(self.unique_dates, dtype=torch.int16)
        self.unique_date, self.inverse_indices, self.counts = torch.unique(self.dates, return_inverse=True, return_counts=True)
        self.n_unique_dates = len(self.unique_date)

        self.X = torch.tensor(self.X.drop(CONFIG.DATE_COL).to_numpy(), dtype=torch.float32, device=self.device)  # type:ignore
        self.y = torch.tensor(self.y.drop(CONFIG.DATE_COL).to_numpy(), dtype=torch.float32, device=self.device)  # type:ignore

    @abstractmethod
    def __getitem__(self, idx):
        """Get item - implemented by subclasses"""
        pass


class SequentialDataset(BaseFinancialDataset):
    """Dataset for sequential models (LSTM, Transformers, CNN)"""

    def __init__(self, X: pl.DataFrame, Y: pl.DataFrame, seq_len: int = CONFIG.SEQ_LEN):
        """
        Sequential dataset for temporal models

        Args:
            data: Preprocessed DataFrame
            target_columns: Target column names (allocation)
            feature_columns: Feature column names
            date_column: Date identifier column
            sequence_length: Number of time steps in sequence
            prediction_horizon: Steps ahead to predict (usually 1)
        """
        self.sequence_length = seq_len

        super().__init__(X, Y)

        self._generate_sequence()

    def _generate_sequence(self):
        self.sequence_x = []
        self.sequence_y = []

        for date in range(self.sequence_length, self.n_unique_dates):
            self.sequence_x.append(self.X[date - self.sequence_length : date])
            self.sequence_y.append(self.y[date - 1])

        self.sequence_x = torch.stack(self.sequence_x)
        self.sequence_y = torch.stack(self.sequence_y)

    def __len__(self):
        return self.n_unique_dates - self.sequence_length

    def __getitem__(self, idx):
        """Get sequence, target, and date_id"""
        continuous_seq = self.sequence_x[idx]  # (seq_len, N_FEATURES)
        if idx + self.sequence_length < self.n_unique_dates:
            target = self.y[idx + self.sequence_length]  # (1,)
        else:
            # For the last sequence, use the last available target
            target = self.y[-1]  # (1,)

        return continuous_seq, target
