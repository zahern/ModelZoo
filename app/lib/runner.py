"""Run a generated script with the engine interpreter and stream its output."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterator


def stream_run(python_exe: str, script_path: str, cwd: str | None = None, env: dict | None = None) -> Iterator[str]:
    """Yield stdout/stderr lines as they arrive from the subprocess (merged, line-buffered)."""
    yield from stream_cmd([python_exe, "-u", str(script_path)], cwd=cwd, env=env)


def stream_cmd(argv: list[str], cwd: str | None = None, env: dict | None = None) -> Iterator[str]:
    """Yield stdout/stderr lines from an arbitrary command (merged, line-buffered)."""
    import os

    full_env = os.environ.copy()
    full_env.setdefault("PYTHONUNBUFFERED", "1")
    full_env.setdefault("PYTHONIOENCODING", "utf-8")
    if env:
        full_env.update(env)

    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=full_env,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            yield line.rstrip("\n")
    finally:
        proc.stdout.close()
        proc.wait()
        yield f"\n[process exited with code {proc.returncode}]"
