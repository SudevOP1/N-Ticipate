"""Phase 7 — the thin ctypes layer over Win32.

Three of the desktop features need the operating system rather than Python:
finding the text caret so the overlay appears next to it, marking the overlay
window as never-activatable so it cannot steal focus, and reading the clipboard
around a paste-based injection. They are collected here rather than duplicated
across :mod:`~nticipate.app.hooks`, :mod:`~nticipate.app.overlay` and
:mod:`~nticipate.app.injector` because all three need the same
``GUITHREADINFO`` struct and the same handle bookkeeping.

Every function in this module is safe to call on a non-Windows machine and safe
to call when the call fails: they return ``None`` (or ``False``) rather than
raising, so the caller always has a fallback path. That is not defensive
padding — the caret query genuinely fails on Electron apps and browsers, which
is the normal case this module has to survive, so "no answer" has to be an
ordinary return value rather than an exception.
"""

from __future__ import annotations

import sys
from typing import NamedTuple

IS_WINDOWS = sys.platform == "win32"

# ---- window styles -------------------------------------------------------
GWL_STYLE = -16
GWL_EXSTYLE = -20

#: The style bit that makes an overlay usable. Without it, clicking the
#: suggestion list -- or in some toolkits merely showing it -- pulls focus out
#: of the app the user is typing in, which is the single bug that makes this
#: class of tool unusable.
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_TOPMOST = 0x00000008

#: Set on a Win32 edit control that masks its input.
ES_PASSWORD = 0x0020
EM_GETPASSWORDCHAR = 0x00D2

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002

SM_CXSCREEN = 0
SM_CYSCREEN = 1

#: Window classes that host their own text rendering and tell Win32 nothing
#: about it. A caret query against one of these succeeds and returns a
#: meaningless rectangle, so they are answered with "unknown" instead.
OPAQUE_WINDOW_CLASSES = frozenset({
    "Chrome_RenderWidgetHostHWND",   # Chrome, Edge, and every Electron app
    "Chrome_WidgetWin_1",
    "MozillaWindowClass",            # Firefox
    "Windows.UI.Core.CoreWindow",    # UWP
    "Windows.UI.Input.InputSite.WindowClass",
})

#: Substrings that, in a window title, make a password field likely even though
#: the control itself is unreadable. Used only for the opaque classes above --
#: it is a guess, and the report says so.
PASSWORD_TITLE_HINTS = ("sign in", "log in", "login", "password", "passphrase",
                        "authenticate", "unlock")


class Point(NamedTuple):
    x: int
    y: int


class Rect(NamedTuple):
    left: int
    top: int
    right: int
    bottom: int


if IS_WINDOWS:  # pragma: no cover - the branch is the platform
    import ctypes
    from ctypes import wintypes

    _user32 = ctypes.windll.user32
    _kernel32 = ctypes.windll.kernel32

    class GUITHREADINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("flags", wintypes.DWORD),
            ("hwndActive", wintypes.HWND),
            ("hwndFocus", wintypes.HWND),
            ("hwndCapture", wintypes.HWND),
            ("hwndMenuOwner", wintypes.HWND),
            ("hwndMoveSize", wintypes.HWND),
            ("hwndCaret", wintypes.HWND),
            ("rcCaret", wintypes.RECT),
        ]

    _user32.GetGUIThreadInfo.argtypes = [wintypes.DWORD, ctypes.POINTER(GUITHREADINFO)]
    _user32.GetGUIThreadInfo.restype = wintypes.BOOL
    _user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    _user32.ClientToScreen.restype = wintypes.BOOL
    _user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
    _user32.GetSystemMetrics.argtypes = [ctypes.c_int]
    _user32.SendMessageW.restype = ctypes.c_longlong

    # GetWindowLongPtr exists only in the 64-bit user32; on 32-bit Windows the
    # plain Long form is the whole API and is already pointer-sized.
    _get_long = getattr(_user32, "GetWindowLongPtrW", None) or _user32.GetWindowLongW
    _set_long = getattr(_user32, "SetWindowLongPtrW", None) or _user32.SetWindowLongW
    _get_long.argtypes = [wintypes.HWND, ctypes.c_int]
    _get_long.restype = ctypes.c_longlong
    _set_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_longlong]
    _set_long.restype = ctypes.c_longlong

    _kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    _kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    _kernel32.GlobalLock.restype = ctypes.c_void_p
    _kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    _user32.GetClipboardData.argtypes = [wintypes.UINT]
    _user32.GetClipboardData.restype = wintypes.HANDLE
    _user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    _user32.SetClipboardData.restype = wintypes.HANDLE


def _gui_thread_info():
    """``GUITHREADINFO`` for the foreground thread, or ``None``."""
    if not IS_WINDOWS:
        return None
    info = GUITHREADINFO()
    info.cbSize = ctypes.sizeof(GUITHREADINFO)
    # Thread id 0 means "whichever thread owns the foreground window", which is
    # exactly the one the user is typing into.
    if not _user32.GetGUIThreadInfo(0, ctypes.byref(info)):
        return None
    return info


def window_class(hwnd) -> str:
    """The class name of a window, or ``''`` if it cannot be read."""
    if not IS_WINDOWS or not hwnd:
        return ""
    buffer = ctypes.create_unicode_buffer(256)
    length = _user32.GetClassNameW(hwnd, buffer, 256)
    return buffer.value if length else ""


def window_title(hwnd) -> str:
    """The title bar text of a window, or ``''``."""
    if not IS_WINDOWS or not hwnd:
        return ""
    buffer = ctypes.create_unicode_buffer(512)
    length = _user32.GetWindowTextW(hwnd, buffer, 512)
    return buffer.value if length else ""


def foreground_title() -> str:
    """The title of the window the user is typing into."""
    info = _gui_thread_info()
    return window_title(info.hwndActive) if info else ""


def screen_size() -> tuple[int, int]:
    """Primary screen size in pixels. ``(0, 0)`` off Windows."""
    if not IS_WINDOWS:
        return (0, 0)
    return (_user32.GetSystemMetrics(SM_CXSCREEN),
            _user32.GetSystemMetrics(SM_CYSCREEN))


def mouse_position() -> Point | None:
    """Cursor position in screen coordinates — the overlay's fallback anchor."""
    if not IS_WINDOWS:
        return None
    point = wintypes.POINT()
    if not _user32.GetCursorPos(ctypes.byref(point)):
        return None
    return Point(point.x, point.y)


def caret_rect() -> Rect | None:
    """The text caret in screen coordinates, or ``None`` if unavailable.

    ``None`` is the expected answer, not an error, for any app that draws its
    own text: browsers, Electron, most UWP. The caller anchors near the mouse
    instead. Windows reports ``rcCaret`` in client coordinates of the window
    that owns the caret, so it has to be translated before it means anything on
    screen.
    """
    info = _gui_thread_info()
    if info is None or not info.hwndCaret:
        return None
    if window_class(info.hwndCaret) in OPAQUE_WINDOW_CLASSES:
        # The call would succeed and return a stale or zeroed rectangle.
        return None
    rect = info.rcCaret
    if rect.right == rect.left and rect.bottom == rect.top:
        return None
    origin = wintypes.POINT(rect.left, rect.top)
    end = wintypes.POINT(rect.right, rect.bottom)
    if not _user32.ClientToScreen(info.hwndCaret, ctypes.byref(origin)):
        return None
    if not _user32.ClientToScreen(info.hwndCaret, ctypes.byref(end)):
        return None
    return Rect(origin.x, origin.y, end.x, end.y)


def is_password_field() -> bool | None:
    """Whether the focused control masks its input.

    Returns ``True``/``False`` for a classic Win32 edit control, whose
    ``ES_PASSWORD`` style is readable, and ``None`` — *unknown* — for the
    browser and Electron windows that render their own inputs and expose
    nothing. The distinction matters: the capture policy in
    :mod:`~nticipate.app.hooks` treats a definite ``True`` as a hard stop and
    an unknown as a cue to fall back to the window title, which is a guess.
    """
    info = _gui_thread_info()
    if info is None:
        return None
    hwnd = info.hwndFocus or info.hwndActive
    if not hwnd:
        return None
    cls = window_class(hwnd)
    if cls in OPAQUE_WINDOW_CLASSES:
        return None
    style = _get_long(hwnd, GWL_STYLE)
    if style & ES_PASSWORD:
        return True
    if "edit" in cls.lower():
        # Some controls set the mask character without the style bit.
        return bool(_user32.SendMessageW(hwnd, EM_GETPASSWORDCHAR, 0, 0))
    return False


def title_suggests_password(title: str | None = None) -> bool:
    """Heuristic used only when :func:`is_password_field` answers ``None``."""
    text = (title if title is not None else foreground_title()).lower()
    return any(hint in text for hint in PASSWORD_TITLE_HINTS)


def set_no_activate(hwnd) -> bool:
    """Mark a window never-activatable, always-on-top, and a tool window.

    Applied to the overlay after Tk has created it. ``WS_EX_NOACTIVATE`` is the
    important one; ``WS_EX_TOOLWINDOW`` additionally keeps the overlay out of
    Alt-Tab, where a one-line suggestion popup has no business being.
    """
    if not IS_WINDOWS or not hwnd:
        return False
    current = _get_long(hwnd, GWL_EXSTYLE)
    wanted = current | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST
    if wanted == current:
        return True
    _set_long(hwnd, GWL_EXSTYLE, wanted)
    # SetWindowLongPtr returns the *previous* value, which is legitimately 0,
    # so the return value cannot be used as a success flag. Read it back.
    return bool(_get_long(hwnd, GWL_EXSTYLE) & WS_EX_NOACTIVATE)


# --------------------------------------------------------------- clipboard

def get_clipboard_text() -> str | None:
    """Current clipboard text, or ``None`` if it holds something else.

    Paired with :func:`set_clipboard_text` so a paste-based injection can put
    the user's clipboard back exactly as it was. Losing someone's clipboard to
    an autocomplete is a bug they notice immediately.
    """
    if not IS_WINDOWS:
        return None
    if not _user32.OpenClipboard(None):
        return None
    try:
        handle = _user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return None
        pointer = _kernel32.GlobalLock(handle)
        if not pointer:
            return None
        try:
            return ctypes.wstring_at(pointer)
        finally:
            _kernel32.GlobalUnlock(handle)
    finally:
        _user32.CloseClipboard()


def set_clipboard_text(text: str | None) -> bool:
    """Replace the clipboard contents. ``None`` empties it."""
    if not IS_WINDOWS:
        return False
    if not _user32.OpenClipboard(None):
        return False
    try:
        _user32.EmptyClipboard()
        if text is None:
            return True
        data = ctypes.create_unicode_buffer(text)
        size = ctypes.sizeof(data)
        handle = _kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
        if not handle:
            return False
        pointer = _kernel32.GlobalLock(handle)
        if not pointer:
            return False
        try:
            ctypes.memmove(pointer, data, size)
        finally:
            _kernel32.GlobalUnlock(handle)
        # Ownership of the handle passes to the clipboard on success; it must
        # not be freed here.
        return bool(_user32.SetClipboardData(CF_UNICODETEXT, handle))
    finally:
        _user32.CloseClipboard()
