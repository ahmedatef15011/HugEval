import json

from acmecli.io_utils import read_urls, write_ndjson_line


def test_read_urls_skips_blanks(tmp_path):
    p = tmp_path / "urls.txt"
    p.write_text("\nhttps://huggingface.co/gpt2\n   \nhttps://huggingface.co/datasets/squad\n")
    urls = list(read_urls(str(p)))
    assert urls == ["https://huggingface.co/gpt2", "https://huggingface.co/datasets/squad"]


def test_write_ndjson_line_writes_valid_json(capsys):
    d = {"a": 1, "b": "x"}
    write_ndjson_line(d)
    out = capsys.readouterr().out.strip()
    parsed = json.loads(out)
    assert parsed == d
