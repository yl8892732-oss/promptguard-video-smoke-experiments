"""PromptGuard public API.

The numerical core is imported lazily so that lightweight submodules such as
``promptguard.benchmarks`` can be used on CPU-only machines without importing
torch, diffusers, or generation code.
"""
from importlib import import_module

__all__ = ["Config", "Router", "grounding", "importance", "select_pattern", "sparse_attention"]


def __getattr__(name):
    if name in __all__:
        value = getattr(import_module(".core", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
