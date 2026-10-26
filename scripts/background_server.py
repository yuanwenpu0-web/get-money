"""Run the local server under Windows Task Scheduler, independently of the chat."""

import logging
import os
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    log_dir = root / "output"
    log_dir.mkdir(exist_ok=True)
    log = (log_dir / "server.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout = log
    sys.stderr = log
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.chdir(root)
    sys.path.insert(0, str(root / "backend"))

    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8010, log_config=None)


if __name__ == "__main__":
    main()
