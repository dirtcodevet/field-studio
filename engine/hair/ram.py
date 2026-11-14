"""This process's peak RAM (standard library only, Apache-2.0). Moved here verbatim from mesh.py (fs-hairclean 11-14), when
engine/hair became mesh-free (P14): the hair modules and their tests read their peak RAM from here.

    peak_mb()   this process's peak working set (MB) on Windows; None elsewhere
"""


def peak_mb():
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (nm, ctypes.c_size_t) for nm in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                                 "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                                 "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        k32, ps = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        p = PMC()
        p.cb = ctypes.sizeof(p)
        ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(p), p.cb)
        return round(p.PeakWorkingSetSize / 2 ** 20)
    except Exception:  # noqa: BLE001
        return None
