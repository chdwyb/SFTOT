import ctypes
import json
import os
from pathlib import Path
import tempfile
import time
import warnings


_REPLACE_DELAYS = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0)


def _replace(source, destination):
    for attempt in range(len(_REPLACE_DELAYS) + 1):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if (not isinstance(error, PermissionError)
                    and getattr(error, "winerror", None) not in (5, 32, 33)):
                raise
            if attempt == len(_REPLACE_DELAYS):
                raise
            time.sleep(_REPLACE_DELAYS[attempt])


def _atomic_write(path, writer):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        _replace(temporary, destination)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as error:
            warnings.warn(f"Could not remove temporary file {temporary}: {error}", RuntimeWarning)
    return True


def atomic_json_write(path, payload, required=False):
    try:
        data = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
        return _atomic_write(path, lambda stream: stream.write(data))
    except Exception as error:
        if required:
            raise
        warnings.warn(f"Could not update optional JSON file {path}: {error}", RuntimeWarning)
        return False


def atomic_torch_save(obj, path):
    import torch
    return _atomic_write(path, lambda stream: torch.save(obj, stream))


def read_json_shared(path):
    if os.name != "nt":
        with Path(path).open(encoding="utf-8-sig") as stream:
            return json.load(stream)
    import msvcrt
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create_file.restype = wintypes.HANDLE
    close_handle = kernel.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    handle = create_file(str(Path(path).resolve()), 0x80000000, 0x00000007,
                         None, 3, 0x80, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except Exception:
        close_handle(handle)
        raise
    with os.fdopen(descriptor, "r", encoding="utf-8-sig") as stream:
        return json.load(stream)


class RunLock:
    def __init__(self, path):
        self.path = Path(path)
        self._stream = None

    def acquire(self):
        if self._stream is not None:
            raise RuntimeError(f"Run lock is already acquired: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            raise RuntimeError(f"Another process holds the run lock: {self.path}") from error
        self._stream = stream
        return self

    def release(self):
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return False
