from pathlib import Path

from acmecli.io_utils import read_urls


def test_read_urls_comma_split(tmp_path: Path):
    p = tmp_path / "urls.txt"
    p.write_text("https://huggingface.co/gpt2, https://huggingface.co/datasets/squad\n")
    urls = list(read_urls(str(p)))
    assert any("gpt2" in u for u in urls)
