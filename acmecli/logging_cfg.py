import logging
import os


def setup_logging() -> None:
    # 0 = silent, 1 = info, 2 = debug (default silent)
    level_map = {"0": logging.CRITICAL + 1, "1": logging.INFO, "2": logging.DEBUG}
    lvl = level_map.get(os.getenv("LOG_LEVEL", "0"), logging.CRITICAL + 1)
    path = os.getenv("LOG_FILE", "acmecli.log")
    logging.basicConfig(
        filename=path, level=lvl, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
