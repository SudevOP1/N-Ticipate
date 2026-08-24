"""Phase 7 — turning an accepted suggestion into keystrokes in someone else's app.

Two things happen on Tab. Working out *what* to insert is arithmetic on the
prefix already typed, and is pure: :func:`plan_injection`. Actually inserting
it means synthesising input into a window this process does not own, which is
:class:`Injector`.

The interesting case in the arithmetic is casing. The predictor returns
truecased words — ``India``, not ``india`` — because that is what the user
wants to read and to keep. But the user may have typed ``ind``, and the
completion of ``ind`` with ``India`` is not ``ia``: appending it would leave
``indIa`` on screen. When the typed prefix does not match the suggestion
character for character, the plan backspaces over the prefix and retypes the
whole word. That costs a few extra keystrokes and is the only way to get the
casing right without silently discarding the truecase map that Phase 1 exists
to build.

Two delivery methods, chosen by ``app.capture.injection_method``:

* ``type`` — pynput types the characters. Universally compatible, and about a
  millisecond per character, which is fine for the 3-8 characters a completion
  usually adds.
* ``clipboard`` — put the text on the clipboard and send Ctrl+V. Constant time
  regardless of length, but it borrows a global resource, so the previous
  clipboard contents are saved and restored. If that restore is skipped the
  app quietly destroys whatever the user had copied, which is a far worse bug
  than a slow paste.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from nticipate.config import get

#: Time to let the target app process the paste before the clipboard is put
#: back. Restoring immediately races the paste and pastes the *old* contents.
#: Not a tunable in config.yaml because it is a property of Windows message
#: delivery rather than of the model.
CLIPBOARD_SETTLE_S = 0.12


@dataclass(frozen=True)
class Injection:
    """What to send: delete ``backspaces`` characters, then type ``text``."""

    backspaces: int
    text: str

    @property
    def keystrokes(self) -> int:
        """Total simulated key presses — the cost side of keystroke savings."""
        return self.backspaces + len(self.text)

    def __bool__(self) -> bool:
        return bool(self.backspaces or self.text)


def plan_injection(prefix: str, word: str, append_space: bool = True) -> Injection:
    """Work out the edit that turns ``prefix`` into ``word``.

    ``prefix`` is what the user has typed of the current word (empty in
    next-word mode). When ``word`` extends it exactly, only the tail is typed;
    otherwise — a casing difference, or a next-word suggestion that shares no
    prefix — the typed characters are deleted first so the result is the word
    the model actually meant.
    """
    tail = " " if append_space else ""
    if not prefix:
        return Injection(0, word + tail)
    if word.startswith(prefix):
        return Injection(0, word[len(prefix):] + tail)
    # Same word, different casing: still a completion, but the visible prefix
    # has to go.
    return Injection(len(prefix), word + tail)


class Injector:
    """Sends an :class:`Injection` to whatever window has focus.

    ``pynput`` is imported on first use rather than at module import, so the
    module and its tests work with the desktop dependencies absent.
    """

    def __init__(self, method: str | None = None, append_space: bool | None = None) -> None:
        self.method = method if method is not None else get(
            "app.capture.injection_method", "type"
        )
        self.append_space = append_space if append_space is not None else get(
            "app.capture.append_space", True
        )
        self._controller = None
        self._lock = threading.Lock()
        #: Counts what was actually sent, for the Phase 8 numbers.
        self.injected_keystrokes = 0
        self.injections = 0

    @staticmethod
    def available() -> bool:
        try:
            import pynput  # noqa: F401
        except Exception:
            return False
        return True

    def _keyboard(self):
        from pynput.keyboard import Controller

        if self._controller is None:
            self._controller = Controller()
        return self._controller

    # ------------------------------------------------------------- delivery

    def plan(self, prefix: str, word: str) -> Injection:
        return plan_injection(prefix, word, self.append_space)

    def accept(self, prefix: str, word: str) -> Injection:
        """Plan and send the insertion of ``word`` over ``prefix``."""
        injection = self.plan(prefix, word)
        self.send(injection)
        return injection

    def send(self, injection: Injection) -> None:
        """Deliver an injection. Serialised — two overlapping accepts interleave
        their keystrokes and produce garbage."""
        if not injection:
            return
        with self._lock:
            keyboard = self._keyboard()
            self._backspace(keyboard, injection.backspaces)
            if self.method == "clipboard" and len(injection.text) > 1:
                self._paste(keyboard, injection.text)
            else:
                keyboard.type(injection.text)
            self.injected_keystrokes += injection.keystrokes
            self.injections += 1

    @staticmethod
    def _backspace(keyboard, count: int) -> None:
        from pynput.keyboard import Key

        for _ in range(count):
            keyboard.press(Key.backspace)
            keyboard.release(Key.backspace)

    def _paste(self, keyboard, text: str) -> None:
        """Ctrl+V the text, then put the user's clipboard back.

        The restore runs in a ``finally``: an exception between the copy and
        the restore would otherwise leave a suggestion sitting in the
        clipboard, which is both surprising and, for anything typed near a
        password field, a small leak.
        """
        from pynput.keyboard import Key

        from nticipate.app import win32

        saved = win32.get_clipboard_text()
        try:
            if not win32.set_clipboard_text(text):
                keyboard.type(text)   # clipboard unavailable; type it instead
                return
            with keyboard.pressed(Key.ctrl):
                keyboard.press("v")
                keyboard.release("v")
            time.sleep(CLIPBOARD_SETTLE_S)
        finally:
            if saved is not None:
                win32.set_clipboard_text(saved)

    def __repr__(self) -> str:
        return (
            f"Injector(method={self.method!r}, injections={self.injections}, "
            f"keystrokes={self.injected_keystrokes})"
        )
