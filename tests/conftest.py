"""
conftest.py — shared pytest fixtures
"""
import sys, os, json
import pytest
import pandas as pd

# ── project root (parent of tests/) ──────────────────────────────────────────
PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


@pytest.fixture(scope="session")
def root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def dataset(root):
    path = os.path.join(root, "data", "marine_water_quality.csv")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def metrics(root):
    path = os.path.join(root, "reports", "model_metrics.json")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def prediction_module():
    """Import prediction module once per session."""
    import prediction as pred
    return pred
