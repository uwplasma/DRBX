"""Backend module name for the test tree."""

from importlib import import_module


BACKEND_PACKAGE = 'drbx.fci_nodal'


def backend_module(name):
    return import_module(f'{BACKEND_PACKAGE}.{name}')
