from __future__ import annotations

import sys

from serve_paper_trading_dashboard import main


if __name__ == "__main__":
    if not any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv):
        sys.argv.extend(["--config", "binary-paper-trading/config_optuned_balanced.yaml"])
    if not any(arg == "--port" or arg.startswith("--port=") for arg in sys.argv):
        sys.argv.extend(["--port", "8098"])
    main()
