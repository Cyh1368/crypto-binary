from __future__ import annotations

import sys

from run_live_paper_trading import LivePaperTrader, ROOT, parse_args


DEFAULT_CONFIG = "binary-paper-trading/config_optuned_balanced.yaml"


if __name__ == "__main__":
    args = parse_args()
    if not any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv):
        args.config = DEFAULT_CONFIG
    trader = LivePaperTrader(ROOT / args.config)
    trader.run(once=args.once)
