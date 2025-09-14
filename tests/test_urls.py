from acmecli.urls import Category, classify


def test_classify_model():
    assert classify("https://huggingface.co/gpt2") is Category.MODEL


def test_classify_dataset():
    assert classify("https://huggingface.co/datasets/squad") is Category.DATASET


def test_classify_code():
    assert classify("https://github.com/user/repo") is Category.CODE
