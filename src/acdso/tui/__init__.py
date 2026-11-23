"""Interactive Terminal UI (TUI) for ACDSO — full-screen Textual app with zero-dependency fallback."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

__all__ = ["launch_tui", "main"]


def launch_tui(initial_file: str | None = None, force_fallback: bool = False) -> int:
    """Launch the interactive ACDSO Terminal UI.

    Uses the full-screen ``Textual`` dashboard when ``textual`` is installed
    (unless ``force_fallback=True``), transparently falling back to the
    zero-dependency ``FallbackTerminalUI`` otherwise.
    """
    from .app import FallbackTerminalUI, has_textual, run_textual_tui

    if not force_fallback and has_textual():
        return run_textual_tui(initial_file=initial_file)
    return FallbackTerminalUI(initial_file=initial_file).run()


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point for ``acdso-tui``."""
    parser = argparse.ArgumentParser(
        prog="acdso-tui",
        description="Launch the interactive ACDSO Terminal UI (Textual with stdlib fallback).",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Optional dataset file (.csv, .xlsx, .parquet, .json) to pre-load.",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Force the zero-dependency interactive terminal fallback instead of Textual.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    return launch_tui(initial_file=args.file, force_fallback=bool(args.fallback))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
