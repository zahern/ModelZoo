"""Locate and probe the Python 'engine' environment that has SearchLibrium,
metacountregressor, and JAX installed. The Streamlit app itself only needs
streamlit + pandas; the heavy model-fitting work always runs in a separate
process using this engine interpreter.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ENGINE_PYTHON = r"Z:\test_runs_tours\code\.venv\Scripts\python.exe"
DEFAULT_ABM_CODE_DIR = r"Z:\test_runs_tours\code"

_PROBE_SCRIPT = r"""
import json, sys
info = {"python_version": sys.version, "packages": {}}
for mod in ("SearchLibrium", "metacountregressor", "jax", "pandas", "numpy"):
    try:
        m = __import__(mod)
        info["packages"][mod] = {
            "ok": True,
            "version": getattr(m, "__version__", "unknown"),
            "location": getattr(m, "__file__", ""),
        }
    except Exception as e:
        info["packages"][mod] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
print(json.dumps(info))
"""


@dataclass
class EngineStatus:
    python_exe: str
    exists: bool
    python_version: str = ""
    packages: dict = field(default_factory=dict)
    error: str = ""

    @property
    def ready(self) -> bool:
        if not self.exists or self.error:
            return False
        sl = self.packages.get("SearchLibrium", {})
        mc = self.packages.get("metacountregressor", {})
        return bool(sl.get("ok") or mc.get("ok"))


def probe_engine(python_exe: str, timeout: int = 120) -> EngineStatus:
    exe = Path(python_exe)
    if not exe.exists():
        return EngineStatus(python_exe=python_exe, exists=False, error="Interpreter not found at this path.")
    try:
        result = subprocess.run(
            [str(exe), "-c", _PROBE_SCRIPT],
            capture_output=True, text=True, timeout=timeout,
        )
    except Exception as e:
        return EngineStatus(python_exe=python_exe, exists=True, error=f"{type(e).__name__}: {e}")

    if result.returncode != 0:
        return EngineStatus(python_exe=python_exe, exists=True, error=result.stderr.strip()[-2000:])

    try:
        data = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as e:
        return EngineStatus(python_exe=python_exe, exists=True, error=f"Could not parse probe output: {e}")

    return EngineStatus(
        python_exe=python_exe,
        exists=True,
        python_version=data.get("python_version", ""),
        packages=data.get("packages", {}),
    )
