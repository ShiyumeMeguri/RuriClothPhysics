import importlib
import pathlib
import subprocess
import sys

_REQUIRED_MODULES = (
    ("numba", "numba"),
    ("numba_cuda", "numba-cuda"),
)

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
