"""Smoke test: the package imports and reports a version."""

from importlib.metadata import version

import sigstack


def test_version_is_set() -> None:
    assert isinstance(sigstack.__version__, str)
    assert sigstack.__version__


def test_version_matches_metadata() -> None:
    # pyproject.toml and __init__.py both carry the version; this catches drift.
    assert sigstack.__version__ == version("sigstack")


def test_core_subpackage_imports() -> None:
    import sigstack.core  # noqa: F401
