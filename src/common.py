from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def environment():
    import numpy
    import pandas
    import scipy
    import sklearn
    import torch
    import torch_geometric
    return {"hostname": socket.gethostname(), "platform": platform.platform(),
            "python": sys.version, "executable": sys.executable,
            "numpy": numpy.__version__, "pandas": pandas.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available()}


@contextlib.contextmanager
def file_lock(path):
    """Prevent duplicate writers; recover only a dead PID on the same host."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        owner = read_json(path)
        if owner["hostname"] != socket.gethostname():
            raise RuntimeError(f"Lock belongs to another host: {path}. Check that job first.")
        try:
            if os.name == "nt":
                # os.kill(pid, 0) on Windows can terminate the target process.
                import ctypes
                from ctypes import wintypes
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.OpenProcess.restype = wintypes.HANDLE
                kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                handle = kernel.OpenProcess(0x1000, False, owner["pid"])
                if not handle:
                    if ctypes.get_last_error() == 87:
                        raise ProcessLookupError()
                    raise PermissionError()
                try:
                    status = wintypes.DWORD()
                    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
                    if not kernel.GetExitCodeProcess(handle, ctypes.byref(status)):
                        raise PermissionError()
                    if status.value != 259:
                        raise ProcessLookupError()
                finally:
                    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
                    kernel.CloseHandle(handle)
            else:
                os.kill(owner["pid"], 0)
        except ProcessLookupError:
            path.unlink()
        except PermissionError:
            raise RuntimeError(f"Cannot verify lock owner: {path}")
        else:
            raise RuntimeError(f"Another process owns {path}: PID {owner['pid']}")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"hostname": socket.gethostname(), "pid": os.getpid()}, handle)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
