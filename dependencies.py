import hashlib
import importlib
import os
import pathlib
import shutil
import subprocess
import sys

_REQUIRED_MODULES = (
    ("numba", "numba"),
    ("numba_cuda", "numba-cuda"),
)

_ROOT = pathlib.Path(__file__).resolve().parent
_KERNEL_CACHE_DIR = _ROOT / "__kernelcache__"
_KERNEL_STAMP = _KERNEL_CACHE_DIR / "source.stamp"
_KERNEL_SOURCE_DIRS = (_ROOT / "cloth_engine_gpu", _ROOT / "cloth_kernel")


def _kernel_source_digest():
    digest = hashlib.sha256()
    for directory in _KERNEL_SOURCE_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            digest.update(str(path.relative_to(_ROOT)).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_kernel_cache():
    digest = _kernel_source_digest()
    if not _KERNEL_STAMP.is_file() or _KERNEL_STAMP.read_text(encoding="utf-8") != digest:
        shutil.rmtree(_KERNEL_CACHE_DIR, ignore_errors=True)
        _KERNEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _KERNEL_STAMP.write_text(digest, encoding="utf-8")
    os.environ["NUMBA_CACHE_DIR"] = str(_KERNEL_CACHE_DIR)
    config = sys.modules.get("numba.core.config")
    if config is not None:
        config.reload_config()


def _importable(module_name):
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    return True


def _missing_packages():
    return [package for module_name, package in _REQUIRED_MODULES if not _importable(module_name)]


def ensure_installed():
    prepare_kernel_cache()
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


prepare_kernel_cache()
