"""Terminal output, ANSI/UTF-8 console setup, and structured progress markers for ACDSO CLI."""

from __future__ import annotations

from contextlib import contextmanager
import io
import json
import os
import sys
from typing import Any


def _is_tty() -> bool:
    """Return True if stdout is connected to an interactive terminal."""
    try:
        return bool(getattr(sys.stdout, "isatty", lambda: False)())
    except Exception:
        return False


def _use_ansi() -> bool:
    """Return True when ANSI terminal styling is safe to emit."""
    return _is_tty() and not os.environ.get("NO_COLOR")


def _configure_console() -> None:
    """Ensure UTF-8 safe console streams and enable Windows VT mode on TTY."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if sys.platform == "win32" and _is_tty():
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass


def _emit_progress(
    phase: int,
    total: int,
    name: str,
    status: str,
    detail: str = "",
) -> None:
    """Flush a structured single-line progress marker for WebUI/TUI consumers."""
    norm_status = status if status in ("running", "done", "error") else "running"
    payload = {
        "phase": int(phase),
        "total": int(total),
        "name": str(name),
        "status": norm_status,
        "detail": "" if detail is None else str(detail),
    }
    print(f"[ACDSO_PROGRESS] {json.dumps(payload, ensure_ascii=False)}", flush=True)


def _render_phase_banner(phase: int, total: int, name: str) -> None:
    """Render a clean terminal phase banner (ANSI on TTY, ASCII when piped)."""
    if _use_ansi():
        print(f"\n\033[1;36m[{phase}/{total}]\033[0m \033[1m{name}\033[0m", flush=True)
    else:
        print(f"\n[{phase}/{total}] {name}", flush=True)


@contextmanager
def capture_console_output():
    """Context manager to capture all stdout and stderr output."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = captured_out = io.StringIO()
    sys.stderr = captured_err = io.StringIO()

    try:
        yield captured_out
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        captured_out.write(captured_err.getvalue())


class DualOutput:
    """File-like object that writes to both console and a buffer."""

    def __init__(self, buffer):
        self.buffer = buffer
        self.console = sys.stdout

    def write(self, text):
        self.console.write(text)
        self.buffer.write(text)

    def flush(self):
        self.console.flush()
        self.buffer.flush()

    def isatty(self):
        return getattr(self.console, "isatty", lambda: False)()


@contextmanager
def capture_and_display_console():
    """Context manager that captures output while still displaying it."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    buffer = io.StringIO()
    sys.stdout = DualOutput(buffer)
    sys.stderr = DualOutput(buffer)

    try:
        yield buffer
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def print_header(title: str, width: int = 80) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subheader(title: str, width: int = 60) -> None:
    """Print a formatted subsection header."""
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print(f"{'-' * width}")


def print_table(headers: list, rows: list, col_widths: list | None = None) -> None:
    """Print a formatted table with headers and rows."""
    if col_widths is None:
        col_widths = [
            max(len(str(h)), max(len(str(row[i])) for row in rows) if rows else 10) + 2
            for i, h in enumerate(headers)
        ]

    header_line = "|".join(str(h).center(w) for h, w in zip(headers, col_widths))
    separator = "┼".join("-" * w for w in col_widths)

    print(f"+{'┬'.join('-' * w for w in col_widths)}+")
    print(f"|{header_line}|")
    print(f"+{separator}┤")

    for row in rows:
        row_line = "|".join(str(cell).center(w) for cell, w in zip(row, col_widths))
        print(f"|{row_line}|")

    print(f"+{'┴'.join('-' * w for w in col_widths)}┘")


def print_dict(
    data: dict[str, Any],
    indent: int = 4,
    max_depth: int = 2,
    current_depth: int = 0,
) -> None:
    """Pretty print a dictionary with controlled depth."""
    prefix = " " * indent * current_depth
    for key, value in data.items():
        if isinstance(value, dict) and current_depth < max_depth:
            print(f"{prefix}• {key}:")
            print_dict(value, indent, max_depth, current_depth + 1)
        elif isinstance(value, list) and len(value) > 0 and current_depth < max_depth:
            print(f"{prefix}• {key}: [{len(value)} items]")
            if len(value) <= 3:
                for item in value:
                    if isinstance(item, dict):
                        print(f"{prefix}    - {item.get('name', item)}")
                    else:
                        print(f"{prefix}    - {item}")
        elif isinstance(value, float):
            print(f"{prefix}• {key}: {value:.4f}")
        else:
            print(f"{prefix}• {key}: {value}")
