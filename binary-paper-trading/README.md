# BTC Live Direction Prediction

BTC-only live predictor for the trained LightGBM direction model.

The live paper-trading server is currently configured to run the `obi-optuna-500` model from `outputs/obi-optuna-500/models/lightgbm_model.pkl`.

It wakes up on 15-minute boundaries, fetches real Kraken Futures data, computes live features, predicts the probability that BTC will be up over the next 15-minute bar, and records every cycle.

No orders are sent to an exchange, and no simulated long/short positions are opened.

## Run Once

```bash
binary-venv/bin/python binary-paper-trading/run_live_paper_trading.py --once
```

## Run Continuously

```bash
binary-venv/bin/python binary-paper-trading/run_live_paper_trading.py
```

## Run Dashboard

```bash
binary-venv/bin/python binary-paper-trading/serve_paper_trading_dashboard.py
```

Then open `http://127.0.0.1:8097`.

The current dashboard server is running at `http://127.0.0.1:8097` and displays the active model name at the top of the page.

## Logs

Outputs are written under `binary-paper-trading/logs/`:

- `paper_trading.log`: human-readable runtime log
- `predictions.csv`: one row per 15-minute prediction
- `price_actions.csv`: OHLCV and current market state per cycle
- `feature_snapshots.jsonl`: complete feature dictionary per cycle
