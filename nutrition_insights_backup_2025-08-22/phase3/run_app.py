# phase3/run_app.py
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path


def _phase3_dir() -> Path:
    """Return absolute path to the phase3 folder (this file's parent)."""
    return Path(__file__).resolve().parent


def _repo_root(start: Path) -> Path:
    """
    Best-effort repo root: look for 'nutrition_insights' folder.
    Falls back to phase3 parent.
    """
    here = start
    ni = here.parent  # .../nutrition_insights
    if (ni / "phase3").exists():
        return ni.parent
    return here.parent


def main() -> int:
    phase3 = _phase3_dir()
    app = phase3 / "app.py"

    if not app.exists():
        print(f"[FATAL] Could not find app.py at: {app}", file=sys.stderr)
        return 2

    # Ensure repo root on sys.path so absolute imports (if any) also work.
    repo = _repo_root(phase3)
    sys.path.insert(0, str(repo.resolve()))

    # Streamlit command
    cmd = [sys.executable, "-m", "streamlit", "run", str(app)]

    # Optional: set a reasonable server.port if you like, otherwise default 8501
    env = os.environ.copy()
    env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")

    print(f"[INFO] Launching Streamlit: {' '.join(cmd)}")
    print(f"[INFO] CWD: {phase3}")
    try:
        return subprocess.call(cmd, cwd=str(phase3), env=env)
    except FileNotFoundError:
        print(
            "[FATAL] Streamlit not found. Install it first:\n\n"
            "  pip install streamlit\n",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())