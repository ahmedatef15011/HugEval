import sys
from typing import Any, Dict, Iterable

import orjson


def read_urls(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield s


def write_ndjson_line(d: Dict[str, Any]) -> None:
    sys.stdout.write(orjson.dumps(d).decode() + "\n")
