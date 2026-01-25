"""LIT configuration for SparDial tests"""

import os
import sys
import lit.formats
import lit.util

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "SPARDIAL"

config.test_format = lit.formats.ShTest(True)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".py", ".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
# When running through CMake, this is set by lit.site.cfg.py
if not hasattr(config, "test_exec_root"):
    config.test_exec_root = config.test_source_root

# excludes: A list of directories to exclude from the testsuite.
config.excludes = [
    "Inputs",
    "Output",
    "CMakeLists.txt",
    "README.txt",
    "lit.cfg.py",
    "lit.site.cfg.py",
    "lit.site.cfg.py.in",
    "__pycache__",
]

# Initialize environment if not already set
if not hasattr(config, "environment"):
    config.environment = {}

# Get Python executable (from CMake if available, otherwise use current interpreter)
if hasattr(config, "python_executable"):
    python_executable = config.python_executable
else:
    python_executable = sys.executable

# Setup PYTHONPATH to include the SparDial python package
if hasattr(config, "spardial_obj_root") and hasattr(config, "spardial_source_root"):
    # Running through CMake
    # config.spardial_obj_root is CMAKE_CURRENT_BINARY_DIR which is build/tools/spardial for in-tree builds
    spardial_root = config.spardial_source_root
    build_root = config.spardial_obj_root
    # Python packages are directly under spardial_obj_root
    build_python_dir = os.path.join(build_root, "python_packages", "spardial")
else:
    # Running directly with lit
    spardial_root = os.path.abspath(os.path.join(config.test_source_root, ".."))
    build_python_dir = os.path.join(
        spardial_root, "build", "tools", "spardial", "python_packages", "spardial"
    )

spardial_python_dir = os.path.join(spardial_root, "python")

# Add build directory to PYTHONPATH (has the compiled extensions and all Python sources)
pythonpath = []
if os.path.exists(build_python_dir):
    pythonpath.append(build_python_dir)
    # Only add environment PYTHONPATH if we found the build directory
    if "PYTHONPATH" in os.environ:
        pythonpath.append(os.environ["PYTHONPATH"])
    config.environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
elif os.path.exists(spardial_python_dir):
    # Fallback to source directory only if build directory doesn't exist
    pythonpath.append(spardial_python_dir)
    if "PYTHONPATH" in os.environ:
        pythonpath.append(os.environ["PYTHONPATH"])
    config.environment["PYTHONPATH"] = os.pathsep.join(pythonpath)

# Add substitutions
config.substitutions.append(("%PYTHON", python_executable))
config.substitutions.append(("%shlibext", ".so"))

# Set up PATH
if "PATH" in os.environ:
    config.environment["PATH"] = os.environ["PATH"]

# Add build/bin and llvm tools to PATH
if hasattr(config, "llvm_tools_dir"):
    # Add LLVM tools directory (includes FileCheck, etc.)
    if "PATH" in config.environment:
        config.environment["PATH"] = config.llvm_tools_dir + os.pathsep + config.environment["PATH"]
    else:
        config.environment["PATH"] = config.llvm_tools_dir

# Add SparDial build/bin to PATH if it exists
# For in-tree builds, bin directory is in the LLVM build root, not under spardial_obj_root
if hasattr(config, "llvm_tools_dir"):
    # llvm_tools_dir already contains the bin directory with all tools, no need to add SparDial bin separately
    pass
elif hasattr(config, "spardial_obj_root"):
    # This shouldn't happen if llvm_tools_dir is set, but handle it anyway
    # Try to find bin directory relative to spardial_obj_root
    # For in-tree builds, go up to LLVM build root
    llvm_build_root = os.path.abspath(os.path.join(config.spardial_obj_root, "..", "..", ".."))
    build_bin_dir = os.path.join(llvm_build_root, "bin")
    if os.path.exists(build_bin_dir):
        if "PATH" in config.environment:
            config.environment["PATH"] = build_bin_dir + os.pathsep + config.environment["PATH"]
        else:
            config.environment["PATH"] = build_bin_dir
else:
    # Running directly with lit, not through CMake
    build_bin_dir = os.path.join(spardial_root, "build", "bin")
    if os.path.exists(build_bin_dir):
        if "PATH" in config.environment:
            config.environment["PATH"] = build_bin_dir + os.pathsep + config.environment["PATH"]
        else:
            config.environment["PATH"] = build_bin_dir
