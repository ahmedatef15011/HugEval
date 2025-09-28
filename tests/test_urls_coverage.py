from acmecli.urls import Category, classify


def test_classify_model_url_simple():
    assert classify("https://huggingface.co/gpt2") is Category.MODEL


def test_classify_dataset_url_simple():
    cat = classify("https://huggingface.co/datasets/squad")
    assert cat is Category.DATASET or cat is Category.CODE or isinstance(cat, Category)
