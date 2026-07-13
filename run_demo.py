#!/usr/bin/env python3
"""
One-command demo launcher for airline network robustness analysis.

Usage:
    python run_demo.py                    # Normal mode
    python run_demo.py --port 8502        # Custom port
"""

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch airline robustness demo")
    parser.add_argument("--port", type=_port, default=8501, help="Streamlit port (default: 8501)")
    args = parser.parse_args(argv)

    root = Path(__file__).parent

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(root / "src" / "app" / "streamlit_app.py"),
        f"--server.port={args.port}",
        "--server.headless=true",
        "--server.address=127.0.0.1",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
    ]

    print(f"Starting Streamlit on port {args.port}...")
    completed = subprocess.run(cmd, cwd=str(root), check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
