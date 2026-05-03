"""Smoke tests for production-safe backend modules."""

import importlib


def test_backend_safe_modules_import():
    for module_name in [
        "trust_call_backend.fusion",
        "trust_call_backend.schemas",
        "trust_call_backend.service_clients",
        "trust_call_backend.metrics_registry",
    ]:
        module = importlib.import_module(module_name)
        assert module is not None
