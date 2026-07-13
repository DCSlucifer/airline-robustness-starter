"""Build and verify the distributable wheel without writing into the repository.

The full datasets and default experiment configuration intentionally remain repository assets.
This smoke test verifies the installable Python packages, dependency metadata, and the console
help path, which does not require those repository-only assets.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DEPENDENCIES = {
    "anthropic",
    "matplotlib",
    "networkx",
    "numpy",
    "openai",
    "pandas",
    "pydantic",
    "pydeck",
    "pyyaml",
    "scipy",
    "streamlit",
}


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = {
        ".git",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".superpowers",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "outputs",
    }
    return {
        name for name in names if name in ignored or name.endswith((".egg-info", ".pyc", ".pyo"))
    }


def _venv_python(venv_dir: Path) -> Path:
    directory = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    return venv_dir / directory / executable


def _console_script(venv_dir: Path) -> Path:
    directory = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    candidates = (
        directory / "airline-robustness.exe",
        directory / "airline-robustness",
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[-1])


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        rendered = subprocess.list2cmdline(command)
        raise RuntimeError(
            f"command failed ({result.returncode}): {rendered}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def _verify_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "src/__init__.py" in names, "wheel does not contain the src package"
        assert "src/app/__init__.py" in names, "wheel does not contain the app package"
        assert any(name.endswith(".dist-info/licenses/LICENSE") for name in names)
        assert any(name.endswith(".dist-info/licenses/DATA_LICENSE.md") for name in names)
        assert not any(name.startswith(("config/", "data/")) for name in names), (
            "repository-only config/data unexpectedly entered the wheel"
        )

        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
        requirements = metadata.get_all("Requires-Dist") or []
        normalized = {
            requirement.split(";", 1)[0]
            .split("[", 1)[0]
            .split(">", 1)[0]
            .split("<", 1)[0]
            .split("=", 1)[0]
            .strip()
            .lower()
            for requirement in requirements
            if "extra ==" not in requirement
        }
        missing = EXPECTED_DEPENDENCIES - normalized
        assert not missing, f"wheel metadata is missing runtime dependencies: {sorted(missing)}"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="airline_package_smoke_") as raw_tmp:
        # Resolve Windows 8.3 short paths so a freshly created venv sees its real location.
        tmp = Path(raw_tmp).resolve()
        checkout = tmp / "checkout"
        wheelhouse = tmp / "wheelhouse"
        venv_dir = tmp / "venv"
        run_dir = tmp / "run"
        wheelhouse.mkdir()
        run_dir.mkdir()
        shutil.copytree(ROOT, checkout, ignore=_copy_ignore)

        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--wheel-dir",
                str(wheelhouse),
                str(checkout),
            ],
            cwd=tmp,
        )
        wheels = list(wheelhouse.glob("*.whl"))
        assert len(wheels) == 1, f"expected one wheel, found: {wheels}"
        wheel = wheels[0]
        _verify_wheel(wheel)

        # Runtime dependencies are already installed by the normal project/CI setup. Reusing
        # them keeps this verification focused and avoids a second dependency download.
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(venv_dir)
        python = _venv_python(venv_dir)
        _run(
            [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
            cwd=run_dir,
        )

        clean_env = os.environ.copy()
        clean_env.pop("PYTHONHOME", None)
        clean_env.pop("PYTHONPATH", None)
        clean_env["PYTHONDONTWRITEBYTECODE"] = "1"
        _run(
            [str(python), "-c", "import src; import src.app; import src.simulate"],
            cwd=run_dir,
            env=clean_env,
        )
        help_text = _run(
            [str(_console_script(venv_dir)), "--help"],
            cwd=run_dir,
            env=clean_env,
        )
        assert "Airline Network Robustness Simulator" in help_text

        print(f"PASS: built and installed {wheel.name}")
        print("PASS: imported src, src.app, and src.simulate outside the checkout")
        print("PASS: airline-robustness --help")
        print("PASS: config/ and data/ remain repository-only")


if __name__ == "__main__":
    main()
