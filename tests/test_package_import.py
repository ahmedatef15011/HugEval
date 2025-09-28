"""Basic package import smoke test."""


def test_package_imports():
    import importlib

    pkg = importlib.import_module("acmecli")
    assert hasattr(pkg, "__version__") or pkg.__name__ == "acmecli"
