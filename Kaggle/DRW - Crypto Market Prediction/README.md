# [DRW - Crypto Market Prediction](https://www.kaggle.com/competitions/drw-crypto-market-prediction)

### 🏆 Leaderboard Rankings
| Category | Rank | Total | Percentile |
|----------|------|-------|------------|
| 🟢 Public  | 198  | 1092  | Top 18%     |
| 🔵 Private | 323  | 1092  | Top 30%     |

---

## 📌 Overview

Develop a model capable of predicting crypto market price movements using high-frequency production data.  
Accurate directional signals derived through quantitative methods can significantly enhance trading strategies and identify market opportunities with greater precision.

---

## 📂 Dataset

### 🧠 Train Set Features
- `bid_qty`: Total quantity buyers are willing to purchase at the best (highest) bid price.
- `ask_qty`: Total quantity sellers are offering at the best (lowest) ask price.
- `buy_qty`: Executed trading quantity at the best ask price (per minute).
- `sell_qty`: Executed trading quantity at the best bid price (per minute).
- `volume`: Total traded volume during the minute.
- `X_{1,...,780}`: Anonymized market features from proprietary sources.
- `label`: Target variable representing anonymized market price movement.

### 🔒 Test Set
- All timestamps are masked, shuffled, and replaced with unique IDs.

---

## 🚀 Approach

### 🔍 Feature Selection
Applied a multi-method approach to rank and select the most predictive features:
- Filters:
  - Low variance
  - High missing values
- Selection Techniques:
  - Pearson correlation
  - Mutual information
  - L1 regularization (Lasso)
  - Tree-based methods (SHAP + stability scores)
- Combines feature scores using a weighted ensemble.

---

### 🧠 Modeling (XGBoost)

Trained and validated models using:
- **Time-based data slices** (e.g., last 80%, 85%, 90%, 95% of data)
- **Time-decay sample weights** (more recent data weighted higher)
- **Outlier-adjusted weights** (to reduce impact of extreme points)
- **K-Fold cross-validation**
- Final predictions averaged across folds (ensemble)

---
