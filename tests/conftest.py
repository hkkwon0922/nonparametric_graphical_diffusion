"""Shared pytest configuration for the DAG-ordering test suite."""
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: expensive test (run with -m slow)")


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT
