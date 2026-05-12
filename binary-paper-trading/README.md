# BTC Live Paper Trading

BTC-only live paper trader for the trained LightGBM direction model.

It wakes up on 15-minute boundaries, fetches real Kraken Futures data, computes live features, predicts the probability that BTC will be up over the next 15-minute bar, and records every cycle.

No orders are sent to an exchange.

## Run Once

```bash
binary-venv/bin/python binary-paper-trading/run_live_paper_trading.py --once
```

## Run Continuously

```bash
binary-venv/bin/python binary-paper-trading/run_live_paper_trading.py
```

## Logs

Outputs are written under `binary-paper-trading/logs/`:

- `paper_trading.log`: human-readable runtime log
- `predictions.csv`: one row per 15-minute prediction
- `price_actions.csv`: OHLCV and current market state per cycle
- `feature_snapshots.jsonl`: complete feature dictionary per cycle
- `paper_positions.csv`: simulated position/equity state per cycle
- `state.json`: latest paper account state for restart continuity

