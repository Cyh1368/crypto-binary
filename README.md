# 15-Minute BTC Futures Direction Prediction with LightGBM

I trained several `LightGBM` models to predict whether BTC would go up or down in 15 minutes, which has tradable markets in [Kalshi](https://kalshi.com/category/crypto/frequency/fifteen_min) and [Polymarket](https://polymarket.com/crypto/15M). 
The current model, [20feat_0515](/outputs/20feat-0515/), is trained on 49,999 15-minute bars as a 17-fold walk-forward LightGBM run using 20 non-raw-price features. I balanced the target inside each training window so that up/down was 50/50. Results are shown below.

I am running [20feat_0515](/outputs/20feat-0515), in real time [here](http://52.208.3.202:8099/). I've previously [tried](https://github.com/Cyh1368/crypto-xgboost/) to predict the exact price in 15 minutes with XGBoost, which failed due to contamination of validation data. I've also found it more feasible to predict the direction only. Until very recently, I have been mistakenly using raw OHLC as features, a fatal flaw in decision trees.

### Obstacles and Concerns
In order of importance,
- How reliable does the signal need to be in this kinds of low signal-to-noise ratio environments? Is 55% a tradable edge? How high should the accuracy / F1 be to say it's a meaningful signal?
- Given a viable signal, how to design a tradable logic?
- What features (see the list below) should I focus on, and what should I not?
- Kalshi's strike price is very hard to predict; it is not the price listed on Kraken, the main data source for the model, nor what is shown on CF Benchmarks' BRTI, which Kalshi claims its prices are based on. I'm not sure whether this will be a big problem, as all that the model needs to predict is the direction, not the exact value of bitcoin. Fortunately, I've found Polymarket's prices to match Kraken's very well. Sadly, Polymarket faces more regulation than Kalshi in the US.
- Assuming a model takes in OBI data at the beginning of the 15-minute contract. I'm concerned that the orderbook may change significantly during the 15 minutes. Thus, I could work on a model that takes in `time-to-next-contract-expiry` as a parameter, and makes a prediction every minute until the contract expires. However, I'm unsure if this is overcomplicating the problem.

## 20feat_0515

Expanding-window walk-forward LightGBM model using 20 non-raw-price features. The run uses 17 validation folds, 50/50 undersampling inside each training window, and 100 Optuna trials optimized for mean walk-forward validation log loss.

Artifacts are in [`outputs/20feat-0515/`](/outputs/20feat-0515/):
- `models/fold_01.pkl` through `models/fold_17.pkl`: per-fold models.
- `models/aggregate_model.pkl`: deployment model retrained on all labeled rows after 50/50 undersampling.
- `models/lightgbm_model.pkl`: existing-loader-compatible alias for the aggregate deployment model.
- `models/recent_5_fold_ensemble.pkl`: convenience ensemble containing folds 13-17 only.
- `predictions/validation_predictions.parquet`: out-of-sample fold predictions.
- `metrics/fold_metrics.csv`: per-fold walk-forward metrics.
- `metrics/focused_model_metrics.csv`: focused fold 17, aggregate, and recent-fold ensemble metrics.

### Metrics

| Metric | 17-fold OOS validation | Fold 17 OOS | Recent 5-fold ensemble on fold 17 | Aggregate on fold 17 |
|---|---:|---:|---:|---:|
| Rows | 5,017 | 319 | 319 | 319 |
| Accuracy | 57.14% | 77.43% | 77.12% | 77.74% |
| Balanced accuracy | 58.17% | 53.60% | 52.87% | 54.85% |
| ROC AUC | 0.6180 | 0.5800 | 0.5782 | 0.5976 |
| Precision | 47.21% | 42.11% | 38.89% | 45.45% |
| Recall | 63.26% | 11.59% | 10.14% | 14.49% |
| F1 | 0.5406 | 0.1818 | 0.1609 | 0.2198 |
| MCC | 0.1603 | 0.1252 | 0.1025 | 0.1575 |
| Log loss | 0.6554 | 0.5433 | 0.5646 | 0.5532 |
| Information coefficient | 0.0259 | 0.0814 | 0.0837 | 0.0952 |

| Training setup | Value |
|---|---:|
| Labeled rows | 49,999 |
| Aggregate balanced rows | 38,954 |
| Fold 17 raw training rows | 49,680 |
| Fold 17 balanced training rows | 38,816 |
| Fold 17 validation period | 2026-05-09 12:30 UTC to 2026-05-12 20:00 UTC |
| Optuna trials | 100 |
| Best mean validation log loss | 0.6559 |

### Plots

![20feat_0515 fold 17 calibration comparison](/outputs/20feat-0515/figures/focused/calibration_fold17_comparison.png)

![20feat_0515 fold 17 ROC comparison](/outputs/20feat-0515/figures/focused/roc_fold17_comparison.png)

![20feat_0515 fold 17 precision-recall comparison](/outputs/20feat-0515/figures/focused/precision_recall_fold17_comparison.png)

![20feat_0515 focused model log loss](/outputs/20feat-0515/figures/focused/log_loss_focused_models.png)

![20feat_0515 aggregate feature importance](/outputs/20feat-0515/figures/focused/feature_importance_aggregate.png)

![20feat_0515 Optuna optimization history](/outputs/20feat-0515/figures/optuna/optimization_history.png)

![20feat_0515 Optuna parameter importance](/outputs/20feat-0515/figures/optuna/param_importance.png)

### Features

| Market activity | Single-bar price shape | Rolling returns | Realized volatility | Parkinson volatility | Regime / statistical state | Session flags |
|---|---|---|---|---|---|---|
| `volume` | `high_low_range` | `rolling_return_3` | `realized_volatility_1` | `parkinson_volatility_1` | `realized_vol_percentile` | `session_asia` |
|  | `close_open_range` | `rolling_return_5` |  | `parkinson_volatility_3` | `rolling_entropy` | `session_europe` |
|  | `vwap_distance` | `rolling_return_15` |  | `parkinson_volatility_5` | `hurst_exponent` |  |
|  | `log_return` | `rolling_return_30` |  | `parkinson_volatility_15` |  |  |
|  |  |  |  | `parkinson_volatility_30` |  |  |
