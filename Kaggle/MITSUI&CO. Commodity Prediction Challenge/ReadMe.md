# MITSUI&CO. Commodity Prediction Challenge

A comprehensive solution for predicting commodity returns using multi-market historical data from London Metal Exchange (LME), Japan Exchange Group (JPX), US Stock markets, and Forex. This project leverages ensemble deep learning techniques to achieve robust long-term forecasting for commodity trading strategies.

## 🎯 Objective

Develop a model capable of predicting future commodity returns using diverse financial market data to optimize trading strategies and manage risk in global commodity markets. The challenge focuses on predicting price-difference series between asset pairs to extract robust price-movement signals.

## 📊 Competition Metric

The evaluation metric is a variant of the Sharpe ratio: **mean Spearman rank correlation** between predictions and targets divided by the **standard deviation** of correlations.

## 🏗️ Architecture Overview

### Data Pipeline
- **Data Sources**: LME, JPX, US Stock, Forex markets
- **Feature Engineering**: 250+ technical indicators and cross-asset relationships
- **Target Processing**: Gaussian rank transformation for stable training
- **Sequence Processing**: Time-series windowing with configurable lookback periods

### Model Architecture
The solution employs an **ensemble approach** combining:

1. **Sequential Models**:
   - GRU-based recurrent networks for temporal pattern capture

2. **Cross-Sectional Models**:
   - Deep MLP for feature interactions

3. **Ensemble Integration**:
   - Learnable weight combination with softmax normalization
   - Entropy regularization to prevent overconfident weighting
   - Dropout layers for regularization

## 🔧 Key Technical Components

### Feature Engineering (`FEATURE_ENGINEERING_FOR_SUBMISSION.py`)
- **Technical Indicators**: Moving averages, On Balance Volume
- **Cross-Asset Features**: Price ratios, correlation features, momentum indicators
- **Market Microstructure**: Volume-based indicators, open interest analysis
- **Statistical Features**: Rolling statistics, volatility measures, skewness, autocorrelation

### Loss Function Innovation (`LOSS_V2.py`)
Custom **multi-objective loss function** combining:
- **Spearman Correlation Loss**: Direct optimization of evaluation metric
- **ListNet Loss**: Learning-to-rank optimization
- **Kendall Tau Loss**: Rank concordance optimization  
- **Pairwise Ranking Loss**: Margin-based ranking
- **Top-K Ranking Loss**: Focus on extreme values
- **MSE Loss**: Stability regularization

### Model Ensemble (`ENSEMBLE_NN.py`)
```python
class ENSEMBLE_NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Sequential models
        self.gru_model = GRUModel(input_dim, hidden_dim, output_dim)
        self.mlp = DeepMLPModel(input_dim, hidden_dim, output_dim)
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(2) / 2)
```

## ⚙️ Configuration & Hyperparameters

Key configuration parameters in `CONFIG.py`:
- **Sequence Length**: 8 time steps (optimized via hyperparameter search)
- **Hidden Dimensions**: 16-64 units depending on model complexity
- **Batch Size**: 32-128 samples
- **Learning Rates**: 0.005-0.01 with separate refit rates
- **Loss Weights**: Balanced combination across ranking objectives

## 🔍 Hyperparameter Optimization

The project includes comprehensive hyperparameter optimization using **Optuna**:
- **Search Space**: Batch size, sequence length, hidden dimensions, learning rates, loss weights
- **Objective**: Validation Sharpe ratio with stability considerations
- **Strategy**: Time series cross-validation with walk-forward analysis
- **Best Configuration**: 
  ```python
  {
      "batch_size": 32,
      "seq_len": 8, 
      "hidden_dim": 16,
      "lr": 0.005,
      "refit_lr": 0.008,
      "spearman_weight": 0.15,
      "listnet_weight": 0.30,
      "pairwise_weight": 0.50
  }
  ```

## 🚀 Training Strategy

### Cross-Validation Approach
- **Time Series Split**: 5-fold walk-forward validation
- **Training Period**: Historical data up to date 1870
- **Validation**: Subsequent periods with realistic trading constraints
- **Model Retraining**: Online adaptation to recent market conditions

### Key Training Features
- **Early Stopping**: Patience-based convergence with validation monitoring
- **Learning Rate Scheduling**: Separate rates for initial training and refit phases
- **Gradient Clipping**: Stability for sequence models
- **Batch Normalization**: Feature scaling and convergence acceleration

## 📈 Performance Metrics

The model achieves:
- **Validation Sharpe Ratio**: ~0.67+ (optimized configuration)
- **Cross-Validation Stability**: Consistent performance across folds
- **Rank Correlation**: Strong Spearman correlation with targets
- **Risk-Adjusted Returns**: Superior information ratio characteristics

## 🛠️ Usage

### Training
```python
from NN_V2 import NN
from ENSEMBLE_NN import ENSEMBLE_NN

# Initialize model
model = NN(
    model=ENSEMBLE_NN(input_dim=len(features), hidden_dim=16, output_dim=424),
    seq_len=8,
    batch_size=32,
    lr=0.005
)

# Train with validation
model.fit(
    train_set=(train_x, train_y),
    val_set=(val_x, val_y),
    verbose=True
)
```

### Prediction
```python
# Generate predictions
predictions = model.predict(test_sequences)

# Calculate Sharpe ratio
sharpe_ratio = rank_correlation_sharpe(targets, predictions)
```

## 📁 Project Structure

```
├── CONFIG.py                    # Configuration parameters
├── ENSEMBLE_NN.py              # Ensemble model architecture  
├── LOSS_V2.py                  # Multi-objective loss functions
├── NN_V2.py                    # Training pipeline and model wrapper
├── FEATURE_ENGINEERING_FOR_SUBMISSION.py  # Feature engineering
├── PREPROCESSOR_V2.py          # Data preprocessing utilities
├── DATASET.py                  # PyTorch dataset implementations
├── SEQUENTIAL_NN_MODEL.py      # Individual sequence models
├── CROSS_SECTIONAL_NN_MODEL.py # Cross-sectional model components
└── NN_modelling_new_loss.ipynb # Hyperparameter optimization notebook
```

## 🔬 Key Innovations

1. **Multi-Market Feature Fusion**: Integration of diverse financial market signals
2. **Ranking-Optimized Loss**: Direct optimization of competition metric
3. **Ensemble Architecture**: Combines temporal and cross-sectional modeling
4. **Online Adaptation**: Model retraining for changing market conditions
5. **Robust Validation**: Time series aware cross-validation strategy

## 🎯 Future Enhancements

- **Alternative Architectures**: Attention mechanisms, graph neural networks
- **Feature Selection**: Automated feature importance analysis
- **Risk Management**: Volatility forecasting integration
- **Market Regime Detection**: Adaptive model selection based on market conditions

## 🤝 Dependencies

- PyTorch 2.0+
- Polars (data processing)
- NumPy, SciPy
- Scikit-learn
- Optuna (hyperparameter optimization)
- TQDM (progress tracking)

