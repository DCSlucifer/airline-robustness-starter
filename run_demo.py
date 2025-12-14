#!/usr/bin/env python3
"""
One-command demo launcher for airline network robustness analysis.

Usage:
    python run_demo.py                    # Normal mode
    python run_demo.py --port 8502        # Custom port
"""
import subprocess
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Launch airline robustness demo")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit port (default: 8501)")
    args = parser.parse_args()

    root = Path(__file__).parent

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(root / "src" / "app" / "streamlit_app.py"),
        f"--server.port={args.port}",
        "--server.headless=true",
    ]

    print(f"Starting Streamlit on port {args.port}...")
    subprocess.run(cmd, cwd=str(root))


if __name__ == "__main__":
    main()
