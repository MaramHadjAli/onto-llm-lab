"""The declared stack imports. These tests do not start a reasoner."""

import importlib

import pytest

MODULES = [
    "rdflib",
    "owlready2",
    "pyshacl",
    "pydantic",
    "typer",
    "pandas",
    "numpy",
    "scipy",
    "matplotlib",
    "httpx",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_stack_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)
