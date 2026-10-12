#!/usr/bin/env python3
"""Validate and select the interpreter used for model training."""

from __future__ import annotations

import importlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile


SUPPORTED_PYTHON_MIN = (3, 10)
SUPPORTED_PYTHON_MAX = (3, 12)
TRAINING_IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "geopandas": "geopandas",
    "shapely": "shapely",
    "osmnx": "osmnx",
    "networkx": "networkx",
    "torch": "torch",
    "scikit-learn": "sklearn",
    "tqdm": "tqdm",
}

# Environment validation may import Matplotlib before any runner configures it.
# Keep this headless and independent of a writable home directory.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "rlevac_matplotlib_cache"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def preferred_python(project_root: str | Path) -> str:
    """Prefer the repository environment, otherwise preserve the active one."""
    root = Path(project_root).resolve()
    candidate = root / ".venv" / "bin" / "python"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return str(Path(sys.executable).absolute())


def training_environment() -> dict:
    """Import the complete training stack and return its runtime identity."""
    version = tuple(sys.version_info[:2])
    if not SUPPORTED_PYTHON_MIN <= version < SUPPORTED_PYTHON_MAX:
        raise RuntimeError(
            "Training requires Python 3.10 or 3.11; "
            f"current interpreter is Python {version[0]}.{version[1]}"
        )

    versions = {}
    failures = {}
    for distribution, module_name in TRAINING_IMPORTS.items():
        try:
            module = importlib.import_module(module_name)
            versions[distribution] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:
            failures[distribution] = f"{type(exc).__name__}: {exc}"
    if failures:
        detail = "; ".join(
            f"{name} ({message})" for name, message in sorted(failures.items())
        )
        raise RuntimeError(
            f"Training environment is incomplete or binary-incompatible: {detail}"
        )

    return {
        "python": platform.python_version(),
        "executable": str(Path(sys.executable).absolute()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": versions,
    }


def validate_python_executable(
    executable: str | Path,
    project_root: str | Path,
) -> str:
    """Fail before a campaign if its subprocess interpreter cannot train."""
    root = Path(project_root).resolve()
    # Preserve a virtual-environment symlink: resolving it to the base Python
    # bypasses pyvenv.cfg and silently drops the environment's packages.
    resolved = Path(executable).expanduser().absolute()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise RuntimeError(f"Training Python is not executable: {resolved}")
    probe = subprocess.run(
        [
            str(resolved),
            "-c",
            (
                "import json; "
                "from RuntimeEnvironment import training_environment; "
                "print(json.dumps(training_environment(), sort_keys=True))"
            ),
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip()
        fallback = preferred_python(root)
        raise RuntimeError(
            f"Training Python failed validation: {resolved}. {detail} "
            f"Use --python {fallback} or rebuild from environment.yml."
        )
    return str(resolved)


if __name__ == "__main__":
    print(json.dumps(training_environment(), indent=2, sort_keys=True))
