import json
import sys
from typing import Iterable, Iterator

from acmecli import main as app


class DummyPool:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def map(self, fn, iterable: Iterable[str]) -> Iterator[dict]:
        for x in iterable:
            yield fn(x)


def test_main_prints_ndjson_for_models(tmp_path, monkeypatch, capsys):
    # create URL file with model + dataset + code; only model should print
    p = tmp_path / "urls.txt"
    p.write_text(
        "https://huggingface.co/gpt2\n"
        "https://huggingface.co/datasets/squad\n"
        "https://github.com/user/repo\n"
    )

    # monkeypatch argv and executor
    monkeypatch.setattr(sys, "argv", ["prog", str(p)])
    monkeypatch.setattr(app.cf, "ProcessPoolExecutor", lambda: DummyPool())

    # run main
    app.main()

    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1  # only 1 MODEL line
    rec = json.loads(out[0])
    assert rec["name"] == "https://huggingface.co/gpt2"
    assert rec["category"] == "MODEL"
    # net_score exists and is in range
    assert 0.0 <= float(rec["net_score"]) <= 1.0
