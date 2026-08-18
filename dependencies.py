"""Bootstraps third-party packages this add-on needs but Blender does not ship.

Blender's bundled Python only carries numpy; ``cloth_engine_gpu`` needs numba and its
CUDA target (numba-cuda). Both are installed straight into Blender's own site-packages
via pip -- the same interpreter that will import them next -- so a fresh Blender
install with this add-on enabled works with no manual setup step.
"""

import importlib
import pathlib
import subprocess
import sys

_REQUIRED_MODULES = (
    ("numba", "numba"),
    ("numba_cuda", "numba-cuda"),
)

# numba's on-disk cache for cloth_engine_gpu.kernels.frame_kernel (cache=True) pickles
# references into the numba/numba-cuda build that compiled it. A cache written by one
# install is not guaranteed loadable by a different install of the same version numbers
# (cloudpickle's module references shift with the underlying build), and a stale entry
# fails hard (ModuleNotFoundError) instead of silently recompiling. Whenever this module
# just installed the dependencies fresh, the cache predates that install and must go.
_KERNEL_CACHE_DIR = pathlib.Path(__file__).resolve().parent / "cloth_engine_gpu" / "__pycache__"


def _clear_stale_kernel_cache():
    if not _KERNEL_CACHE_DIR.is_dir():
        return
    for stale in _KERNEL_CACHE_DIR.glob("*.nb*"):
        stale.unlink(missing_ok=True)


def _importable(module_name):
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    return True


def _missing_packages():
    return [package for module_name, package in _REQUIRED_MODULES if not _importable(module_name)]


def ensure_installed():
    missing = _missing_packages()
    if not missing:
        return
    print("RuriClothPhysics: installing missing dependencies: %s" % ", ".join(missing))
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", *missing], check=True)
    importlib.invalidate_caches()
    still_missing = _missing_packages()
    if still_missing:
        raise RuntimeError(
            "RuriClothPhysics: failed to install required packages: %s"
            % ", ".join(still_missing))
    _clear_stale_kernel_cache()
