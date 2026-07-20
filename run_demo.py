"""Start the server in deterministic local demo mode."""

import os
import sys

os.environ["DEMO_MODE"] = "1"
os.environ.setdefault("AGENT_PORT", "8765")
if len(sys.argv) > 1:
    os.environ["AGENT_PORT"] = sys.argv[1]

from server import main


if __name__ == "__main__":
    main()
