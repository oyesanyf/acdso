"""Web GUI for ACDSO.

A thin Flask front-end over the ``run_acdso.py`` CLI: upload a data file,
pick a target, toggle any of the CLI flags, and watch the run stream back
live.  Nothing here reimplements the pipeline -- every run is the exact
same command line you would have typed.

Start it with ``python start.py`` from the repository root.
"""

from __future__ import annotations

__all__ = ["create_app", "run"]


def run(argv: list[str] | None = None) -> int:
    """Launch the ACDSO Web GUI (entry point for ``acdso-web``)."""
    from pathlib import Path
    import importlib.util

    repo_root = Path(__file__).resolve().parents[3]
    start_py = repo_root / "start.py"
    if start_py.is_file():
        spec = importlib.util.spec_from_file_location("acdso_start", str(start_py))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return int(mod.main(argv))

    from waitress import serve

    app = create_app()
    serve(app, host="127.0.0.1", port=5000, threads=16)
    return 0


def __getattr__(name: str):
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
