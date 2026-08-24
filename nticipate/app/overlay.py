"""Phase 7 — the suggestion overlay.

A frameless, always-on-top, never-focusable window that lists the current
suggestions next to the caret. Tk is used rather than a second GUI toolkit
because it is in the standard library and the tray already needs Pillow; the
window is stripped of everything Tk normally gives it (``overrideredirect``
removes the frame, ``WS_EX_NOACTIVATE`` removes its ability to take focus) so
what is left is a rectangle of text that cannot interfere with typing.

Focus is the whole problem. A suggestion popup that activates when it appears
pulls the caret out of the app being typed into, which drops the next keystroke
and makes the tool worse than useless. Three things prevent that, and all three
are needed:

* ``overrideredirect(True)`` — no frame, no taskbar entry, no window manager
  activation on map;
* ``WS_EX_NOACTIVATE`` via :func:`~nticipate.app.win32.set_no_activate` — the
  window is refused activation even if something asks;
* the overlay is never given a Tk focus or a binding, so it has no reason to
  ask.

Positioning is best-effort by design. :func:`~nticipate.app.win32.caret_rect`
answers for classic Win32 controls (Notepad, most native apps) and returns
``None`` for browsers and Electron, where the mouse position is the fallback
anchor. :func:`anchor_for` implements that choice and is pure, so the fallback
logic is tested without a screen.
"""

from __future__ import annotations

from typing import Sequence

from nticipate.config import get
from nticipate.app import win32

#: Colours per theme: (background, foreground, highlight bg, highlight fg).
THEMES = {
    "dark": ("#1e1e1e", "#d4d4d4", "#0a5a8a", "#ffffff", "#3a3a3a"),
    "light": ("#fbfbfb", "#202020", "#c8e0f4", "#101010", "#c0c0c0"),
}

#: Rough pixel size of one character and one row at the default font size,
#: used to size the window before Tk has laid anything out. Tk can measure the
#: text exactly, but only after the window is mapped -- and mapping it at the
#: wrong size makes it visibly jump.
CHAR_WIDTH = 8
ROW_HEIGHT = 20
PADDING = 10


def format_items(suggestions: Sequence, max_items: int) -> list[str]:
    """The lines to draw: ``1. word`` … numbered so Alt-1..3 is obvious.

    Accepts :class:`~nticipate.predictor.Suggestion` objects or bare strings —
    the editor fallback passes strings.
    """
    lines = []
    for index, item in enumerate(suggestions[:max_items], start=1):
        word = getattr(item, "word", None) or str(item)
        lines.append(f"{index}. {word}")
    return lines


def anchor_for(
    caret: win32.Rect | None,
    mouse: win32.Point | None,
    offset_x: int = 4,
    offset_y: int = 20,
) -> win32.Point | None:
    """Where the overlay's top-left corner should go.

    Below the caret when Windows will say where the caret is, below-right of
    the mouse when it will not, and ``None`` when neither is available — in
    which case the caller does not show the overlay at all, because a popup in
    an arbitrary corner of the screen is worse than no popup.
    """
    if caret is not None:
        return win32.Point(caret.left + offset_x, caret.bottom + offset_y)
    if mouse is not None:
        return win32.Point(mouse.x + offset_x, mouse.y + offset_y)
    return None


def clamp_to_screen(
    x: int, y: int, width: int, height: int,
    screen_width: int, screen_height: int,
) -> win32.Point:
    """Keep the whole window on screen.

    A suggestion list anchored to a caret on the last line of a maximised
    window would otherwise be drawn off the bottom edge, where the taskbar is,
    so it flips above the anchor instead of merely sliding up — sliding would
    put it on top of the text being typed.
    """
    if screen_width <= 0 or screen_height <= 0:
        return win32.Point(x, y)
    if x + width > screen_width:
        x = max(0, screen_width - width)
    if y + height > screen_height:
        # Flip above the anchor: y was caret_bottom + offset, so this lands the
        # window just above the caret line.
        flipped = y - height - ROW_HEIGHT - 2 * PADDING
        y = flipped if flipped >= 0 else max(0, screen_height - height)
    return win32.Point(max(0, x), max(0, y))


class SuggestionOverlay:
    """The popup itself.

    Created against a Tk root the caller owns (the tray app makes one; the
    editor fallback reuses its own), so this class never starts a second Tk
    main loop. All the public methods are safe to call from the keystroke
    thread: each one marshals onto the Tk thread with ``after``, because Tk is
    not thread-safe and a global hook does not run on the main thread.
    """

    def __init__(
        self,
        root=None,
        max_items: int | None = None,
        font_size: int | None = None,
        theme: str | None = None,
        offset_x: int | None = None,
        offset_y: int | None = None,
    ) -> None:
        self.max_items = max_items if max_items is not None else get(
            "app.overlay.max_items", 3
        )
        self.font_size = font_size if font_size is not None else get(
            "app.overlay.font_size", 11
        )
        self.theme = theme if theme is not None else get("app.overlay.theme", "dark")
        self.offset_x = offset_x if offset_x is not None else get("app.overlay.offset_x", 4)
        self.offset_y = offset_y if offset_y is not None else get("app.overlay.offset_y", 20)

        self._root = root
        self._window = None
        self._labels: list = []
        self._items: list[str] = []
        self.selected = 0
        self.visible = False

    # ---------------------------------------------------------------- Tk set-up

    @staticmethod
    def available() -> bool:
        """Whether Tk can be imported. False on a headless build of Python."""
        try:
            import tkinter  # noqa: F401
        except Exception:
            return False
        return True

    def attach(self, root) -> "SuggestionOverlay":
        """Bind to a Tk root. Must happen on the Tk thread."""
        self._root = root
        return self

    @property
    def attached(self) -> bool:
        """Whether there is a Tk root to draw on.

        A *detached* overlay — no root — still tracks which suggestions are up
        and which one is highlighted, it simply draws nothing. That is what
        makes the app's accept/dismiss/navigate logic testable without a
        screen, and it is also the honest behaviour if Tk is unavailable: the
        rest of the pipeline keeps working, silently, rather than crashing the
        keystroke thread on every prediction.
        """
        return self._root is not None

    def _ensure_window(self):
        import tkinter as tk

        if self._window is not None:
            return self._window
        if self._root is None:
            return None

        background, foreground, _, _, border = THEMES.get(self.theme, THEMES["dark"])
        window = tk.Toplevel(self._root)
        window.withdraw()
        window.overrideredirect(True)          # no frame, no taskbar, no activation
        window.attributes("-topmost", True)
        window.configure(background=border)
        frame = tk.Frame(window, background=background, padx=PADDING // 2, pady=2)
        # The 1px border is the frame's parent showing through -- cheaper than a
        # Canvas and it survives the theme switch.
        frame.pack(padx=1, pady=1, fill="both", expand=True)
        self._frame = frame
        self._window = window
        # WS_EX_NOACTIVATE has to be applied after Tk has actually created the
        # HWND, which update_idletasks forces.
        window.update_idletasks()
        win32.set_no_activate(self._hwnd(window))
        return window

    @staticmethod
    def _hwnd(window):
        """The Win32 handle behind a Tk window, or ``None`` off Windows.

        ``winfo_id`` gives Tk's own child window; the extended styles have to
        go on the top-level frame that Windows actually manages, which is what
        ``wm frame`` reports (as a hex string).
        """
        if not win32.IS_WINDOWS:
            return None
        try:
            return int(window.frame(), 16)
        except (ValueError, TypeError, AttributeError):
            return window.winfo_id()

    # ------------------------------------------------------------- rendering

    def _redraw(self) -> None:
        if self._window is None:
            return
        import tkinter as tk

        background, foreground, hl_bg, hl_fg, _ = THEMES.get(self.theme, THEMES["dark"])
        for label in self._labels:
            label.destroy()
        self._labels = []
        for index, line in enumerate(self._items):
            chosen = index == self.selected
            label = tk.Label(
                self._frame,
                text=line,
                anchor="w",
                justify="left",
                font=("Segoe UI", self.font_size),
                background=hl_bg if chosen else background,
                foreground=hl_fg if chosen else foreground,
                padx=6,
                pady=1,
            )
            label.pack(fill="x")
            self._labels.append(label)

    def _place(self) -> bool:
        """Position and map the window. Returns whether it could be shown."""
        window = self._ensure_window()
        if window is None:
            return True          # detached: the state is live, the pixels are not
        width = max((len(line) for line in self._items), default=0) * CHAR_WIDTH + 2 * PADDING
        height = len(self._items) * ROW_HEIGHT + PADDING

        anchor = anchor_for(
            win32.caret_rect(), win32.mouse_position(), self.offset_x, self.offset_y
        )
        if anchor is None:
            return False
        screen_w, screen_h = win32.screen_size()
        if not screen_w:
            screen_w = window.winfo_screenwidth()
            screen_h = window.winfo_screenheight()
        point = clamp_to_screen(anchor.x, anchor.y, width, height, screen_w, screen_h)

        window.geometry(f"{width}x{height}+{point.x}+{point.y}")
        window.deiconify()
        window.lift()
        # Re-assert: some apps steal the topmost slot when they repaint.
        window.attributes("-topmost", True)
        win32.set_no_activate(self._hwnd(window))
        return True

    # ----------------------------------------------------------- public API

    def show(self, suggestions: Sequence) -> None:
        """Display ``suggestions``. Thread-safe."""
        items = format_items(suggestions, self.max_items)
        if not items:
            self.hide()
            return
        self._items = items
        self.selected = min(self.selected, len(items) - 1)
        self._on_tk_thread(self._show_now)

    def _show_now(self) -> None:
        self._ensure_window()
        self._redraw()
        self.visible = self._place()
        if not self.visible:
            self._hide_now()

    def hide(self) -> None:
        """Hide the overlay. Thread-safe, and a no-op if already hidden."""
        self._on_tk_thread(self._hide_now)

    def _hide_now(self) -> None:
        self.visible = False
        self.selected = 0
        if self._window is not None:
            self._window.withdraw()

    def move(self, delta: int) -> None:
        """Move the highlight by ``delta``, wrapping at both ends."""
        if not self._items:
            return
        self.selected = (self.selected + delta) % len(self._items)
        self._on_tk_thread(self._redraw)

    def current(self) -> str | None:
        """The highlighted word, without its ``1. `` prefix."""
        if not self._items or not self.visible:
            return None
        return self._items[self.selected].split(". ", 1)[-1]

    def destroy(self) -> None:
        if self._window is not None:
            self._window.destroy()
            self._window = None
        self.visible = False

    def _on_tk_thread(self, function) -> None:
        """Run ``function`` on the Tk thread.

        The keystroke hook runs on a pynput thread and Tk explodes if touched
        from there, so everything the hook triggers goes through ``after(0)``.
        Without a root — in tests, and in the headless code paths — the call
        happens inline, which keeps the pure logic testable.
        """
        if self._root is None:
            function()
            return
        try:
            self._root.after(0, function)
        except Exception:
            # The root has been destroyed (quit mid-keystroke). Nothing to draw.
            pass

    def __repr__(self) -> str:
        state = "visible" if self.visible else "hidden"
        return f"SuggestionOverlay({state}, items={len(self._items)}, theme={self.theme!r})"
