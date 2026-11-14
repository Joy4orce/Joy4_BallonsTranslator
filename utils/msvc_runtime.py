"""Make sure the process gets the system's MSVC C++ runtime, not Qt's old copy.

PyQt6-Qt6 6.4.2 (pinned, see requirements.txt) ships its own msvcp140.dll in
PyQt6/Qt6/bin, and that copy dates from 2020 (14.26). Windows keeps one module
per DLL name per process, so whichever msvcp140.dll is loaded first is the one
everything else gets. If Qt is imported first, torch's c10.dll — built against a
much newer runtime — is handed the 2020 one and fails to initialise:

    OSError: [WinError 1114] DLL initialization routine failed.
    Error loading "...\\torch\\lib\\c10.dll" or one of its dependencies.

Both apps import Qt long before torch (torch arrives with the first module
load, on a worker thread in the screen translator), so that is the order they
hit. Loading the System32 runtime up front settles it: Qt is happy with a newer
runtime, torch is not happy with an older one.

Import this and call preload_system_msvc_runtime() before the first Qt import.
It needs nothing beyond the standard library, so it is safe that early.
"""
import os
import sys

_RUNTIME_DLLS = (
    'msvcp140.dll',
    'msvcp140_1.dll',
    'msvcp140_2.dll',
    'msvcp140_atomic_wait.dll',
    'msvcp140_codecvt_ids.dll',
)

_done = False


def preload_system_msvc_runtime():
    """Load the System32 MSVC C++ runtime DLLs into the process. No-op off
    Windows, and when a DLL is absent — the failure it guards against then
    surfaces as before, no worse."""
    global _done
    if _done or sys.platform != 'win32':
        return
    _done = True

    import ctypes

    system32 = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32')
    for name in _RUNTIME_DLLS:
        path = os.path.join(system32, name)
        if not os.path.isfile(path):
            continue
        try:
            ctypes.WinDLL(path)
        except OSError:
            pass
