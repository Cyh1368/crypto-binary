# BTC 15-Minute Direction Prediction System

End-to-end BTC direction classification pipeline using only real historical market data.

The pipeline:

- downloads Kraken Futures OHLCV/funding with `ccxt`
- downloads Binance Vision public historical `bookDepth` archives
- stores raw order book arrays in parquet
- builds causal microstructure, derivatives, regime, volatility, and cross-asset feature columns
- trains LightGBM with walk-forward validation
- writes predictions, metrics, model artifacts, feature importance, SHAP importance, calibration, PnL, and regime plots

No synthetic order books, proxy liquidity, fake funding, or inferred depth feeds are generated.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 scripts/run_pipeline.py
```

Use `--force-download` to ignore cached parquet and download real exchange data again.

```bash
python3 scripts/run_pipeline.py --force-download
```

The first run may take a while because it downloads public historical archives.

## Outputs

```text
outputs/
├── models/lightgbm_model.pkl
├── predictions/test_predictions.parquet
├── predictions/probabilities.parquet
├── metrics/classification_metrics.json
├── metrics/financial_metrics.json
├── metrics/regime_metrics.json
└── figures/
    ├── confusion_matrix.png
    ├── prediction_actual_heatmap.png
    ├── feature_importance.png
    ├── shap_importance.png
    ├── calibration_curve.png
    ├── pnl_curve.png
    └── regime_performance.png
```

## Real Data Boundary

Public unauthenticated access is available for Kraken Futures data and Binance Vision book depth archives. Coinbase, Bybit, OKX, and other RTI constituent integrations are represented as explicit ingestion stubs that require real API/archive credentials or exported data. They intentionally raise instead of substituting proxy data.
