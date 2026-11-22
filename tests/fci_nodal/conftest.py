"""Shared fixtures and markers for nodal backend tests."""

from pathlib import Path

import pytest

from tests.fci_nodal.support.backend import backend_module


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    if not any(rule.split(':', 1)[0].strip() == 'slow' for rule in config.getini('markers')):
        config.addinivalue_line('markers', 'slow: excluded from the fast research gate')
    for rule in ('fast: data-free checks in seconds',
                 'pinned_case: opt-in identity checks on a current resolution 16 reference',
                 'rebuild_case: opt-in cold device-to-step resolution 16 checks'):
        config.addinivalue_line('markers', rule)


@pytest.fixture
def runtime_module():
    def module(path="options"):
        return backend_module(path)

    return module


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    test_root = Path(__file__).parent
    for item in items:
        if not item.path.is_relative_to(test_root):
            continue
        if item.get_closest_marker('pinned_case') or item.get_closest_marker('rebuild_case'):
            item.add_marker(pytest.mark.slow)
        else:
            item.add_marker(pytest.mark.fast)
