# 15-Minute BTC Futures Direction Prediction with LightGBM

I trained several `LightGBM` models to predict whether BTC would go up or down in 15 minutes, which has tradable markets in [Kalshi](https://kalshi.com/category/crypto/frequency/fifteen_min) and [Polymarket](https://polymarket.com/crypto/15M). 
The latest model, [optuned-balanced](/outputs/optuned-balanced/) is trained on 50,000 15-minute bars as a 17-fold walk-forward LightGBM ensemble. There are 43 features, including technical indicators, returns, and volatility. I balanced the target so that up/down was 50/50 and had a strict 80-10-10 split. Results are shown below. 

I am running the 2nd latest model, [balanced-50-50](/outputs/balanced_50_50/), in real time [here](http://52.208.3.202:8097/). I've previously [tried](https://github.com/Cyh1368/crypto-xgboost/) to predict the exact price in 15 minutes with XGBoost, which failed due to contamination of validation data. I've also found it more feasible to predict the direction only.

### Obstacles and Concerns
In order of importance,
- Although [balanced-50-50](/outputs/balanced_50_50/) shows a ~55% accuracy in test and validation, its performance is lower than 50% with two days of live data. Could anything be wrong?
- Is 55% a tradable edge? How high should the accuracy / F1 be to say it's a meaningful signal?
- What features (see the list below) should I focus on, and what should I not? I am looking into adding order book imbalance.
- Kalshi's strike price is very hard to predict; it is not the price listed on Kraken, the main data source for the model, nor what is shown on CF Benchmarks' BRTI, which Kalshi claims its prices are based on. I'm not sure whether this will be a big problem, as all that the model needs to predict is the direction, not the exact value of bitcoin. Fortunately, I've found Polymarket's prices to match Kraken's very well. Sadly, Polymarket faces more regulation than Kalshi in the US.
- Assuming a model takes in OBI data at the beginning of the 15-minute contract. I'm concerned that the orderbook may change significantly during the 15 minutes. Thus, I could work on a model that takes in `time-to-next-contract-expiry` as a parameter, and makes a prediction every minute until the contract expires. However, I'm unsure if this is overcomplicating the problem.

## Optuned-balance
### Performance
  | Metric | Test | Validation |
  |---|---:|---:|
  | Accuracy | 54.86% | 55.10% |
  | Balanced accuracy | 54.86% | 55.10% |
  | ROC AUC | 0.5727 | 0.5765 |
  | Precision | 54.11% | 54.16% |
  | Recall | 64.02% | 66.39% |
  | F1 | 0.5865 | 0.5966 |
  | MCC | 0.0990 | 0.1047 |
  | Log loss | 0.6840 | 0.6811 |
  | Information coefficient | 0.0473 | 0.0551 |

  | Confusion Matrix | Pred Down | Pred Up |
  |---|---:|---:|
  | Actual Down | 6,192 | 7,355 |
  | Actual Up | 4,874 | 8,673 |

![Optuned-balance calibration curve](/outputs/optuned-balanced/figures/calibration_curve.png)

### Features

| Raw Market / OHLCV | Funding / Derivatives | Single-Bar Price Shape | Rolling Returns | Rolling Volatility | Realized Volatility | Parkinson Volatility | Market Regime / Statistical State | Session Flags |
|---|---|---|---|---|---|---|---|---|
| `open` | `funding_rate` | `log_return` | `rolling_return_1` | `rolling_volatility_3` | `realized_volatility_1` | `parkinson_volatility_1` | `realized_vol_percentile` | `session_asia` |
| `high` | `funding_change` | `close_open_range` | `rolling_return_3` | `rolling_volatility_5` | `realized_volatility_3` | `parkinson_volatility_3` | `rolling_entropy` | `session_europe` |
| `low` | `funding_zscore` | `high_low_range` | `rolling_return_5` | `rolling_volatility_15` | `realized_volatility_5` | `parkinson_volatility_5` | `hurst_exponent` | `session_us` |
| `close` |  | `vwap` | `rolling_return_15` | `rolling_volatility_30` | `realized_volatility_15` | `parkinson_volatility_15` | `trend_strength` |  |
| `volume` |  | `vwap_distance` | `rolling_return_30` | `rolling_volatility_60` | `realized_volatility_30` | `parkinson_volatility_30` |  |  |
|  |  |  | `rolling_return_60` |  | `realized_volatility_60` | `parkinson_volatility_60` |  |  |

## File Structure

| Path | Description |
|---|---|
| `config/` | YAML configuration for data sources, feature windows, model parameters, walk-forward splits, thresholds, and trading costs. |
| `data/raw/` | Cached raw market data, including exchange OHLCV/funding data and Binance order book depth parquet files. |
| `data/datasets/` | Built model-ready feature datasets, including `BTC_USDT_features.parquet`. |
| `src/data/` | Data download and raw-data construction code for Kraken futures and Binance Vision depth data. |
| `src/features/` | Feature engineering modules for price/technical features, order flow, derivatives, regimes, volatility regimes, cross-asset features, and targets. |
| `src/models/` | Walk-forward split logic and LightGBM training/saving utilities. |
| `src/evaluation/` | Classification metrics, diagnostics, backtest metrics, statistical tests, and plot generation. |
| `src/utils/` | Shared I/O and logging helpers. |
| `scripts/` | Entrypoints for running the BTC pipeline and training archived non-BTC asset models. |
| `outputs/optuned-balanced/` | Latest tuned balanced BTC LightGBM ensemble, including model artifact, feature list, predictions, metrics, validation report, and figures. |
| `outputs/balanced_50_50/` | Previous balanced BTC model artifacts and evaluation outputs. |
| `outputs/models/`, `outputs/metrics/`, `outputs/predictions/`, `outputs/figures/` | Default output locations used by the pipeline when no output subdirectory is supplied. |
| `binary-paper-trading/` | Live BTC paper-trading predictor, runtime config, dashboard/server code, and live logs. |
| `archive/Other Coins/` | Archived balanced LightGBM model runs for BNB, DOGE, ETH, HYPE, SOL, and XRP. |
| `archive/balanced-backtest-live-paper/` | Archived live paper-trading logs/results for the balanced model. |
| `archive/pre_balanced_dataset_paper_trading_results/` | Archived paper-trading results from an earlier pre-balanced dataset run. |
| `references/` | Research notes, planning material, and paper-trading reference artifacts. |
