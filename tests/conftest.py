"""pytest configuration for SparDial tests"""

import sys
import os
from pathlib import Path


def pytest_configure(config):
    """Configure pytest environment"""
    # Add SparDial Python packages to sys.path
    repo_root = Path(__file__).parent.parent
    build_dir = repo_root / 'build'
    python_packages = build_dir / 'tools' / 'spardial' / 'python_packages' / 'spardial'

    if python_packages.exists():
        sys.path.insert(0, str(python_packages))
    else:
        raise RuntimeError(
            f"SparDial Python packages not found at {python_packages}. "
            "Please build the project first with: ninja -j 32 SparDialPythonModules"
        )
