"""Rebuild RunModelZoo.exe from launcher.py.

Usage:
    .venv\\Scripts\\python.exe build_exe.py

Only needs to be re-run if launcher.py changes — the exe just locates and
shells out to .venv + app\\Home.py at runtime, it doesn't bundle Streamlit
or any app code itself (stays a few MB instead of a few hundred).
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "pyinstaller"], check=True)
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--onefile", "--name", "RunModelZoo",
         "--console", "launcher.py"],
        cwd=ROOT, check=True,
    )
    shutil.copy(ROOT / "dist" / "RunModelZoo.exe", ROOT / "RunModelZoo.exe")
    print(f"\nBuilt {ROOT / 'RunModelZoo.exe'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
