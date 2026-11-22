"""Web GUI for ACDSO.

A thin Flask front-end over the ``run_acdso.py`` CLI: upload a data file,
pick a target, toggle any of the CLI flags, and watch the run stream back
live.  Nothing here reimplements the pipeline -- every run is the exact
same command line you would have typed.

Start it with ``python start.py`` from the repository root.
"""

from __future__ import annotations

__all__ = ["create_app"]


def __getattr__(name: str):
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
