# BTC 15-Minute Direction Prediction System — Engineering Specification

# PROJECT GOAL

Build a complete machine learning pipeline from an empty repository that:

1. Downloads ONLY real market data
2. Uses exchange-quality short-horizon microstructure features
3. Trains a LightGBM classifier to predict BTC direction 15 minutes ahead
4. Evaluates statistically and financially
5. Produces:

   * prediction heatmaps
   * confusion matrices
   * feature importance
   * probability calibration plots
   * PnL curves
   * regime diagnostics

The target is:

[
y_t =
\begin{cases}
1 & \text{if } \log(P_{t+15m}/P_t) > 0 \
0 & \text{otherwise}
\end{cases}
]

Prefer directional classification over price regression.

---

# HARD REQUIREMENTS

## ABSOLUTE REQUIREMENTS

* NO proxy data
* NO synthetic order books
* NO fake funding rates
* NO inferred depth curves
* NO simulated liquidity

All features must derive from:

* Binance
* Coinbase
* Kraken
* Bybit
* OKX
* CME (if available)
* CF Benchmarks constituent exchanges

ONLY real historical data.

---

# CF BENCHMARKS ALIGNMENT

The model should preferentially use exchanges contributing to CF Benchmarks / CME CF Bitcoin Real Time Index (RTI):

Priority exchanges:

1. Coinbase
2. Kraken
3. Bitstamp
4. Gemini
5. itBit
6. LMAX Digital

Supplementary futures/orderflow exchanges:

* Binance
* Bybit
* OKX

Reason:
CF RTI methodology prioritizes highly reliable spot venues with anti-manipulation filtering.

The final midprice should therefore be:

[
P_t = \sum_i w_i P_{i,t}
]

using liquidity-weighted cross-exchange spot prices.

---

# REPOSITORY STRUCTURE

```text
btc_direction_model/
│
├── README.md
├── requirements.txt
├── config/
│   ├── data.yaml
│   ├── model.yaml
│   └── feature_config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   └── datasets/
│
├── src/
│   ├── data/
│   │   ├── download_binance.py
│   │   ├── download_coinbase.py
│   │   ├── download_kraken.py
│   │   ├── download_bybit.py
│   │   ├── download_okx.py
│   │   ├── download_orderbook.py
│   │   └── build_reference_price.py
│   │
│   ├── features/
│   │   ├── technical.py
│   │   ├── orderflow.py
│   │   ├── derivatives.py
│   │   ├── regime.py
│   │   ├── cross_asset.py
│   │   ├── volatility.py
│   │   ├── target.py
│   │   └── feature_pipeline.py
│   │
│   ├── models/
│   │   ├── train_lightgbm.py
│   │   ├── inference.py
│   │   └── walk_forward.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── plots.py
│   │   ├── backtest.py
│   │   └── diagnostics.py
│   │
│   └── utils/
│       ├── io.py
│       ├── logging.py
│       └── time.py
│
├── notebooks/
│
├── outputs/
│   ├── figures/
│   ├── models/
│   ├── metrics/
│   └── predictions/
│
└── scripts/
    ├── run_pipeline.py
    ├── train.sh
    └── evaluate.sh
```

---

# REQUIRED DATA SOURCES

# 1. SPOT MARKET DATA

## Coinbase

Use:

* BTC-USD trades
* candles
* order book snapshots

Endpoint:

* Coinbase Advanced API

Purpose:

* RTI-aligned institutional spot anchor

---

## Kraken

Use:

* BTC/USD spot
* Kraken Futures
* funding rate

Purpose:

* RTI constituent
* derivatives information

---

## Bitstamp

Use:

* BTC/USD spot trades
* candles

Purpose:

* RTI constituent

---

## Gemini

Use:

* BTC/USD spot

Purpose:

* RTI constituent

---

## itBit

Use:

* BTC/USD spot

Purpose:

* RTI constituent

---

## LMAX Digital

Use if accessible.

---

# 2. DERIVATIVES DATA

## Binance Futures

Required:

* trades
* funding
* open interest
* liquidations
* depth snapshots

MOST IMPORTANT:
historical order book depth.

Use Binance Vision datasets whenever possible.

---

## Bybit

Required:

* funding
* OI
* liquidations

---

## OKX

Required:

* futures basis
* funding
* OI

---

# 3. CROSS-ASSET DATA

Required:

* ETHUSDT
* SOLUSDT
* Nasdaq futures
* DXY
* VIX
* Gold
* SPY futures

Use real APIs only.

---

# DATA COLLECTION REQUIREMENTS

# ORDER BOOK SNAPSHOTS

This is the MOST IMPORTANT feature set.

Collect:

* top 20 levels minimum
* ideally top 50
* timestamped snapshots

Required cadence:

* 1 minute

Store raw bids/asks arrays.

---

# TRADE FLOW

Required:

* aggressive buy volume
* aggressive sell volume
* signed trade imbalance

Compute using:

* trade side
* maker/taker flags

---

# FUNDING RATES

Store:

* current funding
* funding momentum
* rolling funding z-score

---

# OPEN INTEREST

Store:

* raw OI
* delta OI
* OI acceleration

---

# FEATURE ENGINEERING

# 1. PRICE FEATURES

Compute on:

* 1m
* 3m
* 5m
* 15m
* 30m
* 60m

Features:

```python
log_return
rolling_return
rolling_volatility
realized_volatility
parkinson_volatility
high_low_range
close_open_range
vwap_distance
```

---

# 2. ORDER FLOW FEATURES (CRITICAL)

# Order Book Imbalance

[
OBI =
\frac{
\sum \text{Bid Volume}
----------------------

\sum \text{Ask Volume}
}{
\sum \text{Bid Volume}
+
\sum \text{Ask Volume}
}
]

OBI=\frac{\sum \text{Bid Volume}-\sum \text{Ask Volume}}{\sum \text{Bid Volume}+\sum \text{Ask Volume}}

Compute for:

* top 5 levels
* top 10
* top 20
* top 50

---

# Additional Microstructure Features

Required:

```python
spread_bps
weighted_midprice
microprice
depth_slope
queue_imbalance
trade_imbalance
volume_imbalance
book_pressure
depth_ratio
cancel_rate
trade_arrival_rate
```

---

# MICROPRICE

[
\text{Microprice} =
\frac{
P_{ask} \cdot V_{bid}
+
P_{bid} \cdot V_{ask}
}{
V_{bid}+V_{ask}
}
]

\text{Microprice}=\frac{P_{ask}\cdot V_{bid}+P_{bid}\cdot V_{ask}}{V_{bid}+V_{ask}}

---

# 3. DERIVATIVES FEATURES

Required:

```python
funding_rate
funding_change
funding_zscore
open_interest
oi_change
oi_acceleration
basis
basis_change
liquidation_volume
long_short_ratio
```

---

# 4. REGIME FEATURES

Required:

```python
realized_vol_percentile
rolling_entropy
hurst_exponent
trend_strength
session_asia
session_europe
session_us
```

---

# 5. CROSS-ASSET FEATURES

Required:

```python
eth_return
sol_return
spy_return
qqq_return
vix_return
dxy_return
gold_return
btc_eth_relative_strength
```

---

# TARGET GENERATION

File:
`src/features/target.py`

```python
future_return_15m =
    np.log(close.shift(-15) / close)

target =
    (future_return_15m > 0).astype(int)
```

Alternative:
triple-barrier labeling.

---

# DATASET CONSTRUCTION

# REQUIRED RULES

## NO LEAKAGE

Forbidden:

* centered rolling windows
* future normalization
* global normalization
* future volatility

Allowed:

```python
rolling(window).mean()
```

Not allowed:

```python
rolling(window, center=True)
```

---

# WALK-FORWARD VALIDATION

MANDATORY.

Example:

```text
Train: Jan-Jun
Val: Jul
Test: Aug

Then roll forward.
```

Never random split.

---

# MODEL ARCHITECTURE

# PRIMARY MODEL

Use:

## LightGBMClassifier

Parameters:

```python
LightGBMClassifier(
    objective="binary",
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=8,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42
)
```

Use:

* early stopping
* GPU if available

---

# FEATURE NORMALIZATION

Tree models generally do not require scaling.

However:

* z-score rolling statistics are allowed
* normalization must be causal

---

# TRAINING PIPELINE

# CLASS WEIGHTS

Handle imbalance:

```python
scale_pos_weight =
    negative_samples / positive_samples
```

---

# EARLY STOPPING

Required:

```python
callbacks=[lgb.early_stopping(100)]
```

---

# SAVE OUTPUTS

Save:

* trained model
* feature list
* prediction probabilities
* metrics JSON
* feature importance CSV

---

# EVALUATION

# CLASSIFICATION METRICS

Required:

```python
accuracy
precision
recall
f1
roc_auc
log_loss
mcc
balanced_accuracy
```

---

# FINANCIAL METRICS

Required:

```python
sharpe_ratio
sortino_ratio
profit_factor
max_drawdown
hit_rate
turnover
```

---

# BACKTESTING

Simple strategy:

```python
LONG if p(up) > 0.60
SHORT if p(up) < 0.40
FLAT otherwise
```

Include:

* transaction costs
* slippage

---

# REQUIRED VISUALIZATIONS

# 1. CONFUSION MATRIX HEATMAP

```python
sns.heatmap(confusion_matrix)
```

Must save:
`outputs/figures/confusion_matrix.png`

---

# 2. PREDICTION VS ACTUAL HEATMAP

Build:

* x-axis = predicted probability bucket
* y-axis = realized return bucket

Use:
2D histogram heatmap.

Required output:
`prediction_actual_heatmap.png`

---

# 3. FEATURE IMPORTANCE

Generate:

* gain importance
* SHAP importance

---

# 4. CALIBRATION PLOT

Show:
predicted probability vs realized frequency.

---

# 5. CUMULATIVE PNL

Plot:
strategy cumulative returns.

---

# REQUIRED STATISTICAL TESTS

Compute:

```python
pearson_corr
spearman_corr
mutual_information
information_coefficient
```

---

# ADDITIONAL REQUIRED ANALYSIS

# REGIME ANALYSIS

Evaluate performance by:

* volatility regime
* session
* trend regime

---

# FEATURE STABILITY

Measure:

* rolling feature importance
* PSI (Population Stability Index)

---

# REQUIRED LIBRARIES

```text
pandas
numpy
lightgbm
scikit-learn
ccxt
requests
pyarrow
matplotlib
plotly
seaborn
scipy
statsmodels
shap
ta
```

---

# REQUIREMENTS.TXT

```text
pandas
numpy
lightgbm
scikit-learn
ccxt
requests
pyarrow
matplotlib
plotly
seaborn
scipy
statsmodels
shap
ta
pyyaml
tqdm
joblib
```

---

# IMPORTANT ENGINEERING NOTES

# 1. DO NOT USE PROXY ORDER BOOKS

The current `download_data.py` file creates synthetic order books:

```python
bids_template
asks_template
```

This must be removed entirely.

Only real historical depth snapshots are permitted. 

---

# 2. USE EXISTING REAL DATA LOGIC

The `download_real_data.py` file already correctly:

* downloads Binance Vision depth
* downloads Kraken funding
* aligns timestamps
* builds real bids/asks arrays

This should become the base ingestion engine. 

---

# 3. PRIORITIZE MICROSTRUCTURE FEATURES

Short-horizon alpha mainly comes from:

* order book imbalance
* queue imbalance
* trade imbalance
* spread pressure
* funding dislocations
* OI acceleration

These are more important than classic indicators like RSI.

---

# 4. USE PARQUET EVERYWHERE

All intermediate datasets should be parquet.

---

# 5. STORE RAW SNAPSHOTS

Never only store aggregated features.

Store:

* raw order books
* raw trades
* raw funding

so features can be recomputed later.

---

# FINAL OUTPUTS

The completed system must generate:

```text
outputs/
├── models/
│   └── lightgbm_model.pkl
│
├── predictions/
│   ├── test_predictions.parquet
│   └── probabilities.parquet
│
├── metrics/
│   ├── classification_metrics.json
│   ├── financial_metrics.json
│   └── regime_metrics.json
│
└── figures/
    ├── confusion_matrix.png
    ├── prediction_actual_heatmap.png
    ├── feature_importance.png
    ├── shap_importance.png
    ├── calibration_curve.png
    ├── pnl_curve.png
    └── regime_performance.png
```

---

# SUCCESS CRITERIA

The pipeline is successful if:

1. All data are real
2. No leakage exists
3. Walk-forward validation is implemented
4. Orderflow features dominate feature importance
5. Prediction probabilities are calibrated
6. The backtest includes fees/slippage
7. Statistical diagnostics are exported
8. The model trains end-to-end automatically
9. Outputs are reproducible
10. The entire pipeline runs from a single command:

```bash
python scripts/run_pipeline.py
```
