"""fs-brush 11-12 (a 16 GB RAM budget; p74: her sculpt store grew by doubling, 6.4 KB a brick, the old and new arrays
held together at each growth: 17.8 GB peaks on five body strokes). A Pool is an array of rows that grows WITHOUT copying:
on Windows its address space is reserved once (VirtualAlloc MEM_RESERVE, no memory) and pages are committed as rows are
needed (MEM_COMMIT; the OS zero-fills them), so a growth is 'commit more and take a longer view': the rows already held
stay where they are, nothing is copied and nothing is held twice. Elsewhere (or if reserving fails) it falls back to a
numpy array grown by copying, as before.

Views keep their pool alive (the ctypes buffer under each view holds it), so memory is released only when no view of the
pool is left."""
import ctypes
import os
import sys

import numpy as np

RESERVE_BYTES = int(float(os.environ.get('FS_VMEM_RESERVE_GB', '64')) * 2 ** 30)   # address space per pool (no memory)
CHUNK_BYTES = 64 * 2 ** 10                                                        # commit rounding (the store sets the growth steps)
_K32 = None
_MEM_RESERVE, _MEM_COMMIT, _MEM_RELEASE = 0x2000, 0x1000, 0x8000
_PAGE_NOACCESS, _PAGE_READWRITE = 0x01, 0x04


def _k32():
    global _K32
    if _K32 is None:
        if sys.platform != 'win32' or os.environ.get('FS_VMEM') == '0':
            _K32 = False
        else:
            try:
                from ctypes import wintypes
                k = ctypes.WinDLL('kernel32', use_last_error=True)
                k.VirtualAlloc.restype = ctypes.c_void_p
                k.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
                k.VirtualFree.restype = wintypes.BOOL
                k.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD]
                _K32 = k
            except Exception:
                _K32 = False
    return _K32


class Pool:
    """rows of shape row_shape and dtype, capacity grown by grow(); .array is the (cap, *row_shape) view (cap exactly the
    rows asked for; the pages under it are committed in CHUNK_BYTES steps)"""

    def __init__(self, row_shape, dtype, rows):
        self.row_shape = tuple(int(x) for x in row_shape)
        self.dtype = np.dtype(dtype)
        self.row_bytes = int(np.prod(self.row_shape, dtype=np.int64)) * self.dtype.itemsize if self.row_shape else self.dtype.itemsize
        self.cap = 0
        self._view = None
        self._np = None
        self._base = None
        self._committed = 0
        k = _k32()
        if k:
            base = k.VirtualAlloc(None, RESERVE_BYTES, _MEM_RESERVE, _PAGE_NOACCESS)
            if base:
                self._base = int(base)
        self.grow(rows)

    @property
    def virtual(self):
        return self._base is not None

    def grow(self, rows):
        """capacity at least rows -> True if it changed (the view is then a new, longer one: take .array again)"""
        rows = int(rows)
        if rows <= self.cap:
            return False
        if self._base is not None:
            need = rows * self.row_bytes
            nb = min(((need + CHUNK_BYTES - 1) // CHUNK_BYTES) * CHUNK_BYTES, RESERVE_BYTES)
            if need > RESERVE_BYTES:
                raise MemoryError('sculpt pool past its reserved %.0f GB' % (RESERVE_BYTES / 2 ** 30))
            if nb > self._committed:
                p = _k32().VirtualAlloc(ctypes.c_void_p(self._base + self._committed), nb - self._committed, _MEM_COMMIT,
                                        _PAGE_READWRITE)
                if not p:
                    raise MemoryError('could not commit %.0f MB for the sculpt pool' % ((nb - self._committed) / 2 ** 20))
                self._committed = nb
        else:
            n = np.zeros((rows,) + self.row_shape, self.dtype)
            if self._np is not None:
                n[:self.cap] = self._np
            self._np = n
        self.cap = rows                                  # exactly the rows asked for (the store keeps its arrays equal)
        self._view = None
        return True

    @property
    def array(self):
        if self._view is None:
            if self._base is not None:
                buf = (ctypes.c_char * max(self.cap * self.row_bytes, 1)).from_address(self._base)
                buf._pool = self                            # a view keeps its pool (and so its memory) alive
                self._view = np.frombuffer(buf, self.dtype).reshape((self.cap,) + self.row_shape)
            else:
                self._view = self._np
        return self._view

    def nbytes(self):
        return self._committed if self._base is not None else (self._np.nbytes if self._np is not None else 0)

    def __del__(self):
        try:
            if self._base is not None and _K32:
                _K32.VirtualFree(ctypes.c_void_p(self._base), 0, _MEM_RELEASE)
                self._base = None
        except Exception:
            pass
