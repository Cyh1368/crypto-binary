# BTC Live Direction Prediction

BTC-only live predictor for the trained LightGBM direction model.

The model-specific live paper-trading entrypoints are:

- `obi-optuna-500`: `run_live_obi_optuna_500.py`, dashboard `serve_dashboard_obi_optuna_500.py`, config `config_obi_optuna_500.yaml`, logs `logs_obi_optuna_500/`, dashboard port `8097`.
- `optuned-balanced`: `run_live_optuned_balanced.py`, dashboard `serve_dashboard_optuned_balanced.py`, config `config_optuned_balanced.yaml`, logs `logs_optuned_balanced/`, dashboard port `8098`.

It wakes up on 15-minute boundaries, fetches real Kraken Futures data, computes live features, predicts the probability that BTC will be up over the next 15-minute bar, and records every cycle.

No orders are sent to an exchange, and no simulated long/short positions are opened.

`live_model_folds` controls which walk-forward fold is used for live inference. The default is `latest`, which uses the newest trained fold; set it to `all` only when you intentionally want to average every saved walk-forward fold.

## Run Once

```bash
binary-venv/bin/python binary-paper-trading/run_live_obi_optuna_500.py --once
binary-venv/bin/python binary-paper-trading/run_live_optuned_balanced.py --once
```

## Run Continuously

```bash
binary-venv/bin/python binary-paper-trading/run_live_obi_optuna_500.py
binary-venv/bin/python binary-paper-trading/run_live_optuned_balanced.py
```

## Run Dashboard

```bash
binary-venv/bin/python binary-paper-trading/serve_dashboard_obi_optuna_500.py
binary-venv/bin/python binary-paper-trading/serve_dashboard_optuned_balanced.py
```

Then open:

- `http://127.0.0.1:8097` for `obi-optuna-500`
- `http://127.0.0.1:8098` for `optuned-balanced`

Each dashboard displays the active model name at the top of the page. The OBI dashboard reports OBI warmup progress when the rolling OBI window is incomplete. The `optuned-balanced` model has no OBI features, so it does not require OBI warmup.

## Logs

Outputs are written under each model's configured logs directory:

- `paper_trading.log`: human-readable runtime log
- `predictions.csv`: one row per 15-minute prediction
- `price_actions.csv`: OHLCV and current market state per cycle
- `feature_snapshots.jsonl`: complete feature dictionary per cycle
