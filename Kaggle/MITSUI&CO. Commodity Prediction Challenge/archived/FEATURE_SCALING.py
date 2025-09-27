import polars as pl
import numpy as np
from typing import Dict, List, Optional, Union, Literal
from dataclasses import dataclass
from enum import Enum
import torch


class ScalerType(Enum):
    MIN_MAX = "min_max"
    ROBUST = "robust"
    STANDARD = "standard"
    NONE = "none"


@dataclass
class ScalerConfig:
    """Configuration for a specific scaler"""

    scaler_type: ScalerType
    params: Dict = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


class TORCH_DynamicScaler:
    """
    Dynamic scaler that applies different scaling methods based on column characteristics
    Works with PyTorch tensors and supports GPU computation
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        self.fitted_scalers = {}
        self.column_configs = {}
        self.feature_names = None
        self.device = device if device is not None else torch.device("cpu")

    def _detect_scaler_type(self, data: torch.Tensor) -> ScalerType:
        """
        Auto-detect appropriate scaler based on data characteristics
        """
        # Remove NaN values for statistics calculation
        if data.dtype in [torch.float16, torch.float32, torch.float64]:
            clean_data = data[~torch.isnan(data)]
        else:
            clean_data = data

        if len(clean_data) == 0:
            return ScalerType.NONE

        # Calculate statistics
        q25 = torch.quantile(clean_data, 0.25)
        q75 = torch.quantile(clean_data, 0.75)
        mean_val = torch.mean(clean_data.float())
        std_val = torch.std(clean_data.float())
        min_val = torch.min(clean_data)
        max_val = torch.max(clean_data)

        # Calculate IQR and detect outliers
        iqr = q75 - q25

        # Count outliers using IQR method
        if iqr > 0:
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outlier_count = torch.sum((clean_data < lower_bound) | (clean_data > upper_bound))
            outlier_ratio = (outlier_count.float() / len(clean_data)).item()
        else:
            outlier_ratio = 0

        # Decision logic
        if outlier_ratio > 0.1:  # More than 10% outliers
            return ScalerType.ROBUST
        elif min_val >= 0 and max_val <= 1:
            return ScalerType.NONE  # Already normalized
        elif std_val > 0 and abs(mean_val.item()) < 2 * std_val.item():
            return ScalerType.STANDARD  # Normal-ish distribution
        else:
            return ScalerType.MIN_MAX  # Default to min-max

    def _fit_min_max_scaler(self, data: torch.Tensor, params: Dict) -> Dict:
        """Fit min-max scaler"""
        if data.dtype in [torch.float16, torch.float32, torch.float64]:
            clean_data = data[~torch.isnan(data)]
        else:
            clean_data = data

        min_val = torch.min(clean_data) if len(clean_data) > 0 else torch.tensor(0.0, device=self.device)
        max_val = torch.max(clean_data) if len(clean_data) > 0 else torch.tensor(1.0, device=self.device)
        feature_range = params.get("feature_range", (0, 1))

        range_val = max_val - min_val if max_val != min_val else torch.tensor(1.0, device=self.device)

        return {
            "min_val": min_val.to(self.device),
            "max_val": max_val.to(self.device),
            "range_val": range_val.to(self.device),
            "feature_range": feature_range,
        }

    def _fit_robust_scaler(self, data: torch.Tensor, params: Dict) -> Dict:
        """Fit robust scaler"""
        if data.dtype in [torch.float16, torch.float32, torch.float64]:
            clean_data = data[~torch.isnan(data)]
        else:
            clean_data = data

        if len(clean_data) == 0:
            return {"median_val": torch.tensor(0.0, device=self.device), "iqr": torch.tensor(1.0, device=self.device)}

        median_val = torch.median(clean_data)
        q25 = torch.quantile(clean_data, 0.25)
        q75 = torch.quantile(clean_data, 0.75)
        iqr = q75 - q25 if q75 != q25 else torch.tensor(1.0, device=self.device)

        return {"median_val": median_val.to(self.device), "iqr": iqr.to(self.device)}

    def _fit_standard_scaler(self, data: torch.Tensor, params: Dict) -> Dict:
        """Fit standard scaler"""
        if data.dtype in [torch.float16, torch.float32, torch.float64]:
            clean_data = data[~torch.isnan(data)]
        else:
            clean_data = data.float()

        if len(clean_data) == 0:
            return {"mean_val": torch.tensor(0.0, device=self.device), "std_val": torch.tensor(1.0, device=self.device)}

        mean_val = torch.mean(clean_data.float())
        std_val = torch.std(clean_data.float(), correction=0)

        return {"mean_val": mean_val.to(self.device), "std_val": std_val.to(self.device) if std_val != 0 else torch.tensor(1.0, device=self.device)}

    def fit(
        self,
        X: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        column_configs: Optional[Dict[Union[int, str], Union[ScalerType, ScalerConfig]]] = None,
        auto_detect_columns: Optional[List[Union[int, str]]] = None,
    ) -> "DynamicScaler":
        """
        Fit scalers to the data

        Args:
            X: Input tensor of shape (n_samples, n_features) or (batch_size, seq_len, n_features)
            feature_names: Optional list of feature names
            column_configs: Manual configuration for specific columns (by index or name)
            auto_detect_columns: Columns to auto-detect scaler type for (by index or name)
        """
        if X.dim() not in [2, 3]:
            raise ValueError("Input tensor must be 2D (n_samples, n_features) or 3D (batch_size, seq_len, n_features)")

        # Ensure tensor is on the correct device
        X = X.to(self.device)

        # Handle 3D tensors by flattening the first two dimensions
        original_shape = X.shape
        if X.dim() == 3:
            batch_size, seq_len, n_features = X.shape
            X = X.view(-1, n_features)  # Flatten to (batch_size * seq_len, n_features)
        else:
            n_features = X.shape[1]

        # Store feature names
        if feature_names is not None:
            if len(feature_names) != n_features:
                raise ValueError("Length of feature_names must match number of features")
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(n_features)]

        if column_configs is None:
            column_configs = {}

        if auto_detect_columns is None:
            # Auto-detect for all columns not in manual config
            auto_detect_columns = []
            for i in range(n_features):
                if i not in column_configs and self.feature_names[i] not in column_configs:
                    auto_detect_columns.append(i)

        # Convert string column names to indices
        def _get_column_index(col):
            if isinstance(col, int):
                return col
            elif isinstance(col, str) and col in self.feature_names:
                return self.feature_names.index(col)
            else:
                raise ValueError(f"Column {col} not found")

        # Process manual configurations
        processed_configs = {}
        for col, config in column_configs.items():
            col_idx = _get_column_index(col)
            if isinstance(config, ScalerType):
                config = ScalerConfig(config)
            processed_configs[col_idx] = config

        # Process auto-detect columns
        for col in auto_detect_columns:
            col_idx = _get_column_index(col)
            if col_idx < n_features:
                scaler_type = self._detect_scaler_type(X[:, col_idx])
                processed_configs[col_idx] = ScalerConfig(scaler_type)

        self.column_configs = processed_configs

        # Fit scalers
        for col_idx, config in self.column_configs.items():
            if col_idx >= n_features:
                continue

            data = X[:, col_idx]

            if config.scaler_type == ScalerType.MIN_MAX:
                self.fitted_scalers[col_idx] = self._fit_min_max_scaler(data, config.params)
            elif config.scaler_type == ScalerType.ROBUST:
                self.fitted_scalers[col_idx] = self._fit_robust_scaler(data, config.params)
            elif config.scaler_type == ScalerType.STANDARD:
                self.fitted_scalers[col_idx] = self._fit_standard_scaler(data, config.params)
            elif config.scaler_type == ScalerType.NONE:
                self.fitted_scalers[col_idx] = {}

        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Apply fitted scalers to transform the data"""

        if not self.fitted_scalers:
            raise ValueError("Scaler not fitted. Call fit() first.")

        if X.dim() not in [2, 3]:
            raise ValueError("Input tensor must be 2D (n_samples, n_features) or 3D (batch_size, seq_len, n_features)")

        # Ensure tensor is on the correct device and proper dtype
        X = X.to(self.device)
        original_shape = X.shape

        # Handle 3D tensors
        if X.dim() == 3:
            batch_size, seq_len, n_features = X.shape
            X = X.view(-1, n_features)

        # Start with a copy of the original tensor
        result = X.clone().float()

        # Apply transformations
        for col_idx, config in self.column_configs.items():
            if col_idx >= result.shape[1]:
                continue

            scaler_params = self.fitted_scalers[col_idx]

            if config.scaler_type == ScalerType.MIN_MAX:
                min_val = scaler_params["min_val"]
                range_val = scaler_params["range_val"]
                feature_range = scaler_params["feature_range"]

                result[:, col_idx] = (result[:, col_idx] - min_val) / range_val * (feature_range[1] - feature_range[0]) + feature_range[0]

            elif config.scaler_type == ScalerType.ROBUST:
                median_val = scaler_params["median_val"]
                iqr = scaler_params["iqr"]

                result[:, col_idx] = (result[:, col_idx] - median_val) / iqr

            elif config.scaler_type == ScalerType.STANDARD:
                mean_val = scaler_params["mean_val"]
                std_val = scaler_params["std_val"]

                result[:, col_idx] = (result[:, col_idx] - mean_val) / std_val

            # ScalerType.NONE: No transformation needed

        # Reshape back to original shape if necessary
        if len(original_shape) == 3:
            result = result.view(original_shape)

        return result

    def fit_transform(
        self,
        X: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        column_configs: Optional[Dict[Union[int, str], Union[ScalerType, ScalerConfig]]] = None,
        auto_detect_columns: Optional[List[Union[int, str]]] = None,
    ) -> torch.Tensor:
        """Fit and transform in one step"""
        return self.fit(X, feature_names, column_configs, auto_detect_columns).transform(X)


class NP_DynamicScaler:
    """
    Dynamic scaler that applies different scaling methods based on column characteristics
    Works with numpy arrays
    """

    def __init__(self):
        self.fitted_scalers = {}
        self.column_configs = {}
        self.feature_names = None

    def _detect_scaler_type(self, data: np.ndarray) -> ScalerType:
        """
        Auto-detect appropriate scaler based on data characteristics
        """
        # Remove NaN values for statistics calculation
        clean_data = data[~np.isnan(data)] if data.dtype.kind in "fc" else data

        if len(clean_data) == 0:
            return ScalerType.NONE

        # Calculate statistics
        q25 = np.percentile(clean_data, 25)
        q75 = np.percentile(clean_data, 75)
        mean_val = np.mean(clean_data)
        std_val = np.std(clean_data)
        min_val = np.min(clean_data)
        max_val = np.max(clean_data)

        # Calculate IQR and detect outliers
        iqr = q75 - q25

        # Count outliers using IQR method
        if iqr > 0:
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outlier_count = np.sum((clean_data < lower_bound) | (clean_data > upper_bound))
            outlier_ratio = outlier_count / len(clean_data)
        else:
            outlier_ratio = 0

        # Decision logic
        if outlier_ratio > 0.1:  # More than 10% outliers
            return ScalerType.ROBUST
        elif min_val >= 0 and max_val <= 1:
            return ScalerType.NONE  # Already normalized
        elif std_val > 0 and abs(mean_val) < 2 * std_val:
            return ScalerType.STANDARD  # Normal-ish distribution
        else:
            return ScalerType.MIN_MAX  # Default to min-max

    def _fit_min_max_scaler(self, data: np.ndarray, params: Dict) -> Dict:
        """Fit min-max scaler"""
        clean_data = data[~np.isnan(data)] if data.dtype.kind in "fc" else data

        min_val = np.min(clean_data) if len(clean_data) > 0 else 0
        max_val = np.max(clean_data) if len(clean_data) > 0 else 1
        feature_range = params.get("feature_range", (0, 1))

        range_val = max_val - min_val if max_val != min_val else 1

        return {"min_val": min_val, "max_val": max_val, "range_val": range_val, "feature_range": feature_range}

    def _fit_robust_scaler(self, data: np.ndarray, params: Dict) -> Dict:
        """Fit robust scaler"""
        clean_data = data[~np.isnan(data)] if data.dtype.kind in "fc" else data

        if len(clean_data) == 0:
            return {"median_val": 0, "iqr": 1}

        median_val = np.median(clean_data)
        q25 = np.percentile(clean_data, 25)
        q75 = np.percentile(clean_data, 75)
        iqr = q75 - q25 if q75 != q25 else 1

        return {"median_val": median_val, "iqr": iqr}

    def _fit_standard_scaler(self, data: np.ndarray, params: Dict) -> Dict:
        """Fit standard scaler"""
        clean_data = data[~np.isnan(data)] if data.dtype.kind in "fc" else data

        if len(clean_data) == 0:
            return {"mean_val": 0, "std_val": 1}

        mean_val = np.mean(clean_data)
        std_val = np.std(clean_data)

        return {"mean_val": mean_val, "std_val": std_val if std_val != 0 else 1}

    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        column_configs: Optional[Dict[Union[int, str], Union[ScalerType, ScalerConfig]]] = None,
        auto_detect_columns: Optional[List[Union[int, str]]] = None,
    ) -> "DynamicScaler":
        """
        Fit scalers to the data

        Args:
            X: Input array of shape (n_samples, n_features)
            feature_names: Optional list of feature names
            column_configs: Manual configuration for specific columns (by index or name)
            auto_detect_columns: Columns to auto-detect scaler type for (by index or name)
        """
        if X.ndim != 2:
            raise ValueError("Input array must be 2D")

        n_samples, n_features = X.shape

        # Store feature names
        if feature_names is not None:
            if len(feature_names) != n_features:
                raise ValueError("Length of feature_names must match number of features")
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(n_features)]

        if column_configs is None:
            column_configs = {}

        if auto_detect_columns is None:
            # Auto-detect for all columns not in manual config
            auto_detect_columns = []
            for i in range(n_features):
                if i not in column_configs and self.feature_names[i] not in column_configs:
                    auto_detect_columns.append(i)

        # Convert string column names to indices
        def _get_column_index(col):
            if isinstance(col, int):
                return col
            elif isinstance(col, str) and col in self.feature_names:
                return self.feature_names.index(col)
            else:
                raise ValueError(f"Column {col} not found")

        # Process manual configurations
        processed_configs = {}
        for col, config in column_configs.items():
            col_idx = _get_column_index(col)
            if isinstance(config, ScalerType):
                config = ScalerConfig(config)
            processed_configs[col_idx] = config

        # Process auto-detect columns
        for col in auto_detect_columns:
            col_idx = _get_column_index(col)
            if col_idx < n_features:
                scaler_type = self._detect_scaler_type(X[:, col_idx])
                processed_configs[col_idx] = ScalerConfig(scaler_type)

        self.column_configs = processed_configs

        # Fit scalers
        for col_idx, config in self.column_configs.items():
            if col_idx >= n_features:
                continue

            data = X[:, col_idx]

            if config.scaler_type == ScalerType.MIN_MAX:
                self.fitted_scalers[col_idx] = self._fit_min_max_scaler(data, config.params)
            elif config.scaler_type == ScalerType.ROBUST:
                self.fitted_scalers[col_idx] = self._fit_robust_scaler(data, config.params)
            elif config.scaler_type == ScalerType.STANDARD:
                self.fitted_scalers[col_idx] = self._fit_standard_scaler(data, config.params)
            elif config.scaler_type == ScalerType.NONE:
                self.fitted_scalers[col_idx] = {}

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted scalers to transform the data"""

        if not self.fitted_scalers:
            raise ValueError("Scaler not fitted. Call fit() first.")

        if X.ndim != 2:
            raise ValueError("Input array must be 2D")

        # Start with a copy of the original array
        result = X.copy().astype(np.float64)

        # Apply transformations
        for col_idx, config in self.column_configs.items():
            if col_idx >= X.shape[1]:
                continue

            scaler_params = self.fitted_scalers[col_idx]

            if config.scaler_type == ScalerType.MIN_MAX:
                min_val = scaler_params["min_val"]
                range_val = scaler_params["range_val"]
                feature_range = scaler_params["feature_range"]

                result[:, col_idx] = (result[:, col_idx] - min_val) / range_val * (feature_range[1] - feature_range[0]) + feature_range[0]

            elif config.scaler_type == ScalerType.ROBUST:
                median_val = scaler_params["median_val"]
                iqr = scaler_params["iqr"]

                result[:, col_idx] = (result[:, col_idx] - median_val) / iqr

            elif config.scaler_type == ScalerType.STANDARD:
                mean_val = scaler_params["mean_val"]
                std_val = scaler_params["std_val"]

                result[:, col_idx] = (result[:, col_idx] - mean_val) / std_val

            # ScalerType.NONE: No transformation needed

        return result

    def fit_transform(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        column_configs: Optional[Dict[Union[int, str], Union[ScalerType, ScalerConfig]]] = None,
        auto_detect_columns: Optional[List[Union[int, str]]] = None,
    ) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X, feature_names, column_configs, auto_detect_columns).transform(X)


class PL_DynamicScaler:
    """
    Dynamic scaler that applies different scaling methods based on column characteristics
    """

    def __init__(self):
        self.fitted_scalers = {}
        self.column_configs = {}

    def _detect_scaler_type(self, series: pl.Series) -> ScalerType:
        """
        Auto-detect appropriate scaler based on data characteristics
        """
        # Get basic statistics
        stats = series.describe()

        # Extract values safely
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        mean_val = series.mean()
        std_val = series.std()
        min_val = series.min()
        max_val = series.max()

        # Calculate IQR and detect outliers
        iqr = q75 - q25 if (q25 is not None and q75 is not None) else 0

        # Count outliers using IQR method
        if iqr > 0:
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outlier_count = series.filter((series < lower_bound) | (series > upper_bound)).len()
            outlier_ratio = outlier_count / len(series)
        else:
            outlier_ratio = 0

        # Decision logic
        if outlier_ratio > 0.1:  # More than 10% outliers
            return ScalerType.ROBUST
        elif min_val is not None and max_val is not None and min_val >= 0 and max_val <= 1:
            return ScalerType.NONE  # Already normalized
        elif std_val is not None and std_val > 0 and abs(mean_val or 0) < 2 * std_val:
            return ScalerType.STANDARD  # Normal-ish distribution
        else:
            return ScalerType.MIN_MAX  # Default to min-max

    def _fit_min_max_scaler(self, series: pl.Series, params: Dict) -> Dict:
        """Fit min-max scaler"""
        min_val = series.min()
        max_val = series.max()
        feature_range = params.get("feature_range", (0, 1))

        return {"min_val": min_val, "max_val": max_val, "range_val": max_val - min_val if max_val != min_val else 1, "feature_range": feature_range}

    def _fit_robust_scaler(self, series: pl.Series, params: Dict) -> Dict:
        """Fit robust scaler"""
        median_val = series.median()
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25 if q75 != q25 else 1

        return {"median_val": median_val, "iqr": iqr}

    def _fit_standard_scaler(self, series: pl.Series, params: Dict) -> Dict:
        """Fit standard scaler"""
        mean_val = series.mean()
        std_val = series.std()

        return {"mean_val": mean_val, "std_val": std_val if std_val != 0 else 1}

    def fit(
        self,
        df: pl.DataFrame,
        column_configs: Optional[Dict[str, Union[ScalerType, ScalerConfig]]] = None,
        auto_detect_columns: Optional[List[str]] = None,
    ) -> "DynamicScaler":
        """
        Fit scalers to the data

        Args:
            df: Input dataframe
            column_configs: Manual configuration for specific columns
            auto_detect_columns: Columns to auto-detect scaler type for
        """

        if column_configs is None:
            column_configs = {}

        if auto_detect_columns is None:
            # Auto-detect for all numeric columns not in manual config
            numeric_cols = [col for col in df.columns if df[col].dtype.is_numeric()]
            auto_detect_columns = [col for col in numeric_cols if col not in column_configs]

        # Process manual configurations
        for col, config in column_configs.items():
            if isinstance(config, ScalerType):
                config = ScalerConfig(config)
            self.column_configs[col] = config

        # Auto-detect configurations
        for col in auto_detect_columns:
            if col in df.columns:
                scaler_type = self._detect_scaler_type(df[col])
                self.column_configs[col] = ScalerConfig(scaler_type)

        # Fit scalers
        for col, config in self.column_configs.items():
            if col not in df.columns:
                continue

            series = df[col]

            if config.scaler_type == ScalerType.MIN_MAX:
                self.fitted_scalers[col] = self._fit_min_max_scaler(series, config.params)
            elif config.scaler_type == ScalerType.ROBUST:
                self.fitted_scalers[col] = self._fit_robust_scaler(series, config.params)
            elif config.scaler_type == ScalerType.STANDARD:
                self.fitted_scalers[col] = self._fit_standard_scaler(series, config.params)
            elif config.scaler_type == ScalerType.NONE:
                self.fitted_scalers[col] = {}

        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply fitted scalers to transform the data"""

        if not self.fitted_scalers:
            raise ValueError("Scaler not fitted. Call fit() first.")

        # Start with the original dataframe
        result = df.clone()

        # Apply transformations
        expressions = []

        for col, config in self.column_configs.items():
            if col not in df.columns:
                continue

            scaler_params = self.fitted_scalers[col]

            if config.scaler_type == ScalerType.MIN_MAX:
                min_val = scaler_params["min_val"]
                range_val = scaler_params["range_val"]
                feature_range = scaler_params["feature_range"]

                expr = ((pl.col(col) - min_val) / range_val * (feature_range[1] - feature_range[0]) + feature_range[0]).alias(col)

            elif config.scaler_type == ScalerType.ROBUST:
                median_val = scaler_params["median_val"]
                iqr = scaler_params["iqr"]

                expr = ((pl.col(col) - median_val) / iqr).alias(col)

            elif config.scaler_type == ScalerType.STANDARD:
                mean_val = scaler_params["mean_val"]
                std_val = scaler_params["std_val"]

                expr = ((pl.col(col) - mean_val) / std_val).alias(col)

            elif config.scaler_type == ScalerType.NONE:
                expr = pl.col(col)  # No transformation

            expressions.append(expr)

        # Keep non-scaled columns as they are
        scaled_cols = set(self.column_configs.keys())
        for col in df.columns:
            if col not in scaled_cols:
                expressions.append(pl.col(col))

        return result.with_columns(expressions)

    def fit_transform(
        self,
        df: pl.DataFrame,
        column_configs: Optional[Dict[str, Union[ScalerType, ScalerConfig]]] = None,
        auto_detect_columns: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """Fit and transform in one step"""
        return self.fit(df, column_configs, auto_detect_columns).transform(df)
