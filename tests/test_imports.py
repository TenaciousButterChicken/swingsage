"""Smoke test: every top-level package imports without crashing on Mac/CPU."""

import importlib

import pytest

MODULES = [
    "capture",
    "capture.config",
    "capture.vtrack_watcher",
    "inference",
    "analytics",
    "coach",
    "ui",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)
