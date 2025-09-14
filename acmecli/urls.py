from enum import Enum


class Category(str, Enum):
    MODEL = "MODEL"
    DATASET = "DATASET"
    CODE = "CODE"


def classify(url: str) -> Category:
    u = url.lower()
    if "huggingface.co/datasets" in u:
        return Category.DATASET
    if "huggingface.co" in u:
        return Category.MODEL
    return Category.CODE
