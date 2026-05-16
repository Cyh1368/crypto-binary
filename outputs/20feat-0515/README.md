# 20feat-0515

Expanding-window walk-forward LightGBM model using the requested 20 features.

Artifacts:
- `models/fold_01.pkl` through `models/fold_17.pkl`: per-fold models.
- `models/aggregate_model.pkl`: deployment model retrained on all labeled rows after 50/50 undersampling.
- `models/lightgbm_model.pkl`: existing-loader-compatible alias for the aggregate deployment model.
- `models/recent_5_fold_ensemble.pkl`: convenience ensemble containing folds 13-17 only.
- `metrics/optuna_best.json`: best 100-trial Optuna hyperparameters by mean walk-forward validation log loss.
- `metrics/optuna_trials.csv`: full Optuna trial table.
- `predictions/validation_predictions.parquet`: out-of-sample fold predictions.
- `metrics/fold_metrics.csv`: per-fold log-loss-led metrics.
- `metrics/focused_model_metrics.csv`: log-loss-led stats for fold 17, aggregate, and folds 13-17 ensemble.
- `figures/optuna/`: Optuna optimization history and parameter importance plots.
- `figures/focused/`: ROC, precision-recall, calibration, probability, confusion, and importance plots.
