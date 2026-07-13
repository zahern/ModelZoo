"""Double-clickable launcher for the ModelZoo Streamlit app.

When built into RunModelZoo.exe (see build_exe.py), this starts Streamlit
using the .venv sitting next to the exe and opens the app in the default
browser. Also runs fine unfrozen: `python launcher.py`.
"""
import os
import subprocess
import sys
import threading
import time
import webbrowser


def _base_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    base_dir = _base_dir()
    venv_python = os.path.join(base_dir, ".venv", "Scripts", "python.exe")
    app_path = os.path.join(base_dir, "app", "Home.py")

    if not os.path.exists(venv_python):
        print(f"Could not find the app's virtual environment at:\n  {venv_python}")
        print("Run this from the ModelZoo repo root after setting up .venv "
              "(see README.md > Getting started).")
        input("\nPress Enter to close...")
        return 1

    if not os.path.exists(app_path):
        print(f"Could not find app\\Home.py next to this launcher (expected {app_path}).")
        input("\nPress Enter to close...")
        return 1

    print("Starting ModelZoo...")
    print(f"  venv:  {venv_python}")
    print(f"  app:   {app_path}")
    print("\nA browser tab will open shortly. Close this window to stop the app.\n")

    threading.Timer(3.0, lambda: webbrowser.open("http://localhost:8501")).start()

    proc = subprocess.Popen(
        [venv_python, "-m", "streamlit", "run", app_path,
         "--server.headless", "true", "--server.port", "8501"],
        cwd=base_dir,
    )
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 0


if __name__ == "__main__":
    sys.exit(main())
