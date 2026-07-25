"""Smoke test: the environment and package layout are importable.

Exists so the post-edit hook has something real to run before M1 lands.
"""

import importlib

import pytest

MODULES = [
    "src.sim",
    "src.episodes",
    "src.causal",
    "src.twin",
    "src.forecast",
    "src.advisor",
    "src.evidence",
    "src.feedback",
    "src.ui",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_core_dependencies_present() -> None:
    for dep in ("numpy", "pandas", "pyarrow", "scipy", "sklearn", "statsmodels"):
        importlib.import_module(dep)
