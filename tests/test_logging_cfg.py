from __future__ import annotations

import logging
import os
from pathlib import Path


def setup_logging() -> None:
    # 0=silent, 1=info, 2=debug
    level_map = {"0": logging.CRITICAL + 1, "1": logging.INFO, "2": logging.DEBUG}
    lvl = level_map.get(os.getenv("LOG_LEVEL", "0"), logging.CRITICAL + 1)

    path = os.getenv("LOG_FILE", "acmecli.log")
    # make sure the folder exists (important for tmp paths in tests)
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Force reconfigure even if handlers already exist (pytest often adds one)
    logging.basicConfig(
        filename=path,
        level=lvl,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
