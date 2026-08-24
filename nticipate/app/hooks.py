"""Phase 7 — keystroke capture: the context buffer, the debouncer, the hook.

The three pieces are deliberately separated by how testable they are:

* :class:`ContextBuffer` and :class:`KeyRouter` are pure Python with no
  dependency on ``pynput`` or on Windows. They hold all the logic that can be
  got wrong — what resets the sentence, what Tab means, when capture must stop
  — so ``tests/test_app.py`` can drive them directly.
* :class:`Debouncer` is a timer. It needs real time to test but nothing else.
* :class:`KeystrokeHook` is the ``pynput`` adapter, and does as little as
  possible: normalise a key event and hand it to the router.

``pynput`` is imported lazily, inside :meth:`KeystrokeHook.start`, for the same
reason Phase 1 has a regex tokenizer fallback — the rest of the module, and its
tests, must work on a machine where the desktop dependencies were never
installed.

Privacy is a structural property of this file, not a setting it checks. The
buffer is a plain string in memory, capped at
``app.capture.max_buffer_chars``, cleared at every sentence end, and it has no
serialisation method at all: there is no ``save``, no ``to_dict``, and its
``__repr__`` reports lengths rather than contents, so a stray log line or a
crash traceback cannot spill what the user typed.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

from nticipate.config import get
from nticipate.app import win32

#: Characters that end a sentence, and therefore the prediction context. The
#: danda and double danda are here because the same buffer serves the Hindi
#: model from Phase 5 -- Devanagari does not use the full stop.
DEFAULT_SENTENCE_END = ".!?।॥"

#: Key names, as :func:`normalize_key` reports them, that pynput gives us for
#: keys with no character. Only the ones the app reacts to are listed; anything
#: else is ignored, which is the safe default for a global hook.
NAVIGATION_KEYS = frozenset({
    "left", "right", "up", "down", "home", "end", "page_up", "page_down",
})

#: Windows key-down messages, as they reach a low-level keyboard hook. Only
#: these two are looked at by the accept filter: swallowing the key-up of a
#: suppressed key-down is unnecessary (no application saw the press) and
#: swallowing key-ups in general strands modifier state in other processes.
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104

#: Virtual key codes for the keys that can be configured as ``accept``. pynput
#: knows these too, but only once it is imported, and the filter has to answer
#: from inside the hook procedure where an import would be a bad idea.
ACCEPT_VK_CODES = {"tab": 0x09, "enter": 0x0D, "return": 0x0D, "space": 0x20}

#: Spellings people write in ``config.yaml`` mapped to every name pynput might
#: report for the same physical key.
KEY_ALIASES: dict[str, frozenset[str]] = {
    "esc": frozenset({"esc", "escape"}),
    "escape": frozenset({"esc", "escape"}),
    "enter": frozenset({"enter", "return"}),
    "return": frozenset({"enter", "return"}),
    "tab": frozenset({"tab"}),
    "space": frozenset({"space"}),
}


class Action(str, Enum):
    """What the app should do about a key, decided by :class:`KeyRouter`."""

    IGNORE = "ignore"        # nothing to do; suggestions stay as they are
    UPDATE = "update"        # buffer changed, ask for a new prediction
    ACCEPT = "accept"        # insert the highlighted suggestion
    DISMISS = "dismiss"      # hide the overlay, keep the buffer
    TOGGLE = "toggle"        # enable/disable the whole app
    NEXT = "next"            # move the highlight down the list
    PREVIOUS = "previous"    # move the highlight up


@dataclass(frozen=True)
class KeyEvent:
    """A key press, normalised away from ``pynput``'s two key types.

    Exactly one of ``char`` and ``name`` is meaningful: a printable key has a
    character, a functional key has a name like ``"tab"`` or ``"enter"``.
    """

    char: str | None = None
    name: str | None = None
    ctrl: bool = False
    alt: bool = False

    @property
    def is_printable(self) -> bool:
        return self.char is not None and self.char.isprintable()


@dataclass(frozen=True)
class Hotkey:
    """A modifier combination plus one key, e.g. ``<ctrl>+<alt>+n``."""

    key: str
    ctrl: bool = False
    alt: bool = False

    def matches(self, event: KeyEvent) -> bool:
        if event.ctrl != self.ctrl or event.alt != self.alt:
            return False
        if event.char is not None:
            return event.char.lower() == self.key
        return (event.name or "").lower() in KEY_ALIASES.get(self.key, {self.key})

    def __str__(self) -> str:
        parts = (["<ctrl>"] if self.ctrl else []) + (["<alt>"] if self.alt else [])
        return "+".join(parts + [self.key])


def parse_hotkey(spec: str | None) -> Hotkey | None:
    """Parse pynput's hotkey spelling, ``<ctrl>+<alt>+n``.

    Only ctrl and alt are honoured. Shift is deliberately not a modifier here:
    the app has to see shifted characters as *characters* — capital letters are
    the whole reason the truecase map exists — so treating Shift as part of a
    chord would swallow ordinary typing.
    """
    if not spec:
        return None
    ctrl = alt = False
    key = ""
    for part in str(spec).split("+"):
        token = part.strip().strip("<>").lower()
        if token in {"ctrl", "ctrl_l", "ctrl_r"}:
            ctrl = True
        elif token in {"alt", "alt_l", "alt_r", "alt_gr"}:
            alt = True
        elif token:
            key = token
    return Hotkey(key=key, ctrl=ctrl, alt=alt) if key else None


class ContextBuffer:
    """The current sentence, in memory, and nothing else.

    Holds the text since the last sentence boundary and splits it on demand
    into the two things :meth:`~nticipate.predictor.Predictor.predict` wants:
    the preceding context words and the partial word under the caret. The
    split itself is delegated to the predictor's own
    :meth:`~nticipate.predictor.Predictor.split_buffer` in the running app, so
    the tokenizer that trained the model is the tokenizer that reads the
    buffer; the light-weight :attr:`words` / :attr:`prefix` here exist for the
    reset logic and for tests.
    """

    def __init__(
        self,
        max_chars: int | None = None,
        max_words: int | None = None,
        sentence_end: str | None = None,
    ) -> None:
        self.max_chars = max_chars if max_chars is not None else get(
            "app.capture.max_buffer_chars", 200
        )
        self.max_words = max_words if max_words is not None else get(
            "app.capture.max_context_words", 2
        )
        self.sentence_end = (
            sentence_end if sentence_end is not None
            else get("app.capture.sentence_end_chars", DEFAULT_SENTENCE_END)
        )
        self._text = ""

    # -------------------------------------------------------------- content

    @property
    def text(self) -> str:
        return self._text

    def __len__(self) -> int:
        return len(self._text)

    def __bool__(self) -> bool:
        return bool(self._text)

    @property
    def words(self) -> list[str]:
        """Whitespace-split words before the caret, at most ``max_words``.

        Deliberately *not* the model's tokenizer: this is used to decide when
        there is enough context to bother predicting, and a whitespace split is
        both faster and unambiguous about where the current word starts.
        """
        parts = self._text.split()
        if self._text and not self._text[-1].isspace():
            parts = parts[:-1]
        return parts[-self.max_words:] if self.max_words else []

    @property
    def prefix(self) -> str:
        """The partial word under the caret; empty after a space."""
        if not self._text or self._text[-1].isspace():
            return ""
        return self._text.split()[-1] if self._text.split() else ""

    # --------------------------------------------------------------- update

    def feed(self, char: str) -> bool:
        """Add one typed character. Returns whether the context survived it.

        ``False`` means the character ended the sentence and the buffer was
        cleared — the caller should hide any suggestion rather than re-predict
        against an empty context.
        """
        if not char:
            return True
        if char in self.sentence_end:
            self.reset()
            return False
        self._text += char
        if len(self._text) > self.max_chars:
            # Ring-buffer the *front* away. The tail is what predicts; the
            # head is only there because the sentence has not ended yet.
            self._text = self._text[-self.max_chars:]
        return True

    def feed_text(self, text: str) -> bool:
        survived = True
        for char in text:
            survived = self.feed(char)
        return survived

    def backspace(self, count: int = 1) -> None:
        """Delete backwards.

        A backspace past the start of the buffer cannot be undone — the text it
        would reveal was cleared at the last sentence end and was never stored
        anywhere. The buffer simply empties, and the next space re-seeds it.
        """
        if count > 0:
            self._text = self._text[:-count] if count < len(self._text) else ""

    def reset(self) -> None:
        """Drop everything. Called on Enter, focus change, click, sentence end."""
        self._text = ""

    def __repr__(self) -> str:
        # Lengths, never contents: this object is the one thing in the process
        # that holds what the user is typing.
        return f"ContextBuffer(chars={len(self._text)}, words={len(self._text.split())})"


class KeyRouter:
    """Turns key events into :class:`Action` values, and updates the buffer.

    The only genuinely interesting decision here is Tab. Tab is configured both
    as the accept key and as a context-reset key, which is not a contradiction:
    while a suggestion is showing, Tab means "take it"; with nothing showing it
    is an ordinary Tab, which moves focus to another field and therefore ends
    the sentence. The router needs to know whether the overlay is up, which is
    why :attr:`suggesting` is set by the app rather than inferred.
    """

    def __init__(
        self,
        buffer: ContextBuffer | None = None,
        accept: str | None = None,
        dismiss: str | None = None,
        toggle: str | None = None,
        reset_keys: Sequence[str] | None = None,
    ) -> None:
        self.buffer = buffer if buffer is not None else ContextBuffer()
        self.accept = (accept if accept is not None else get("app.hotkeys.accept", "tab")).lower()
        self.dismiss = (
            dismiss if dismiss is not None else get("app.hotkeys.dismiss", "esc")
        ).lower()
        self.toggle = parse_hotkey(
            toggle if toggle is not None else get("app.hotkeys.toggle", "<ctrl>+<alt>+n")
        )
        reset = reset_keys if reset_keys is not None else get(
            "app.capture.context_reset_keys", ["enter", "tab", "escape"]
        )
        self.reset_keys = {str(k).lower() for k in reset}
        #: Set by the app whenever the overlay is shown or hidden.
        self.suggesting = False

    def _resets(self, binding: str) -> bool:
        """Whether a configured binding also appears in ``context_reset_keys``."""
        return bool(KEY_ALIASES.get(binding, {binding}) & self.reset_keys)

    @staticmethod
    def _matches(event: KeyEvent, binding: str) -> bool:
        """Whether an event is the configured key.

        ``esc``/``escape`` and ``enter``/``return`` are both spellings people
        write in config files, and pynput reports only one of each pair.
        """
        name = (event.name or "").lower()
        return bool(name) and name in KEY_ALIASES.get(binding, {binding})

    def route(self, event: KeyEvent) -> Action:
        """Update the buffer for ``event`` and say what the app should do."""
        name = (event.name or "").lower()

        if self.toggle is not None and self.toggle.matches(event):
            return Action.TOGGLE

        if self._matches(event, self.dismiss):
            if self._resets(self.dismiss):
                self.buffer.reset()
            return Action.DISMISS

        if self._matches(event, self.accept) and self.suggesting:
            # The accepted text is fed back into the buffer by the app, after
            # the injector has actually typed it.
            return Action.ACCEPT

        if name in self.reset_keys or name in {"enter", "return"}:
            self.buffer.reset()
            return Action.DISMISS

        if name in NAVIGATION_KEYS:
            # Arrows move through the list while it is up; otherwise the caret
            # has moved somewhere the buffer cannot account for, so the context
            # is no longer trustworthy and is dropped.
            if self.suggesting and name in {"down", "up"}:
                return Action.NEXT if name == "down" else Action.PREVIOUS
            self.buffer.reset()
            return Action.DISMISS

        if name in {"backspace", "delete"}:
            if name == "backspace":
                self.buffer.backspace()
                return Action.UPDATE
            # Forward delete removes text to the *right* of the caret, which
            # the buffer does not model. The context to the left is unchanged.
            return Action.IGNORE

        if event.ctrl or event.alt:
            # A shortcut, not typing. Ctrl+V and friends change the document in
            # ways the buffer cannot see, so the safe move is to forget.
            self.buffer.reset()
            return Action.DISMISS

        if event.is_printable and event.char is not None:
            survived = self.buffer.feed(event.char)
            return Action.UPDATE if survived else Action.DISMISS

        if name == "space":
            self.buffer.feed(" ")
            return Action.UPDATE

        return Action.IGNORE


class Debouncer:
    """Run a callback once the keystrokes stop, not once per keystroke.

    Prediction costs a few milliseconds; a fast typist produces keystrokes
    faster than that, and every superseded prediction is wasted work whose
    result would be stale before it was drawn. Each call to :meth:`schedule`
    cancels the pending timer, so exactly one prediction runs per burst of
    typing, ``delay_ms`` after the last key.
    """

    def __init__(self, delay_ms: int | None = None) -> None:
        self.delay_ms = delay_ms if delay_ms is not None else get(
            "prediction.latency.debounce_ms", 50
        )
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    @property
    def pending(self) -> bool:
        with self._lock:
            return self._timer is not None and self._timer.is_alive()

    def schedule(self, callback: Callable[[], None]) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.delay_ms / 1000.0, callback)
            self._timer.daemon = True
            self._timer.start()

    def cancel(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


class CapturePolicy:
    """Decides whether capture is allowed to run at all, right now.

    Separate from the router because it answers a different question — not
    "what does this key mean" but "should this key be looked at". The password
    rule is a hard requirement from the plan, so the failure mode matters: a
    definite ``True`` from Win32 blocks capture, and an *unknown* (browsers,
    Electron) falls back to the window title, which is a genuine guess and is
    reported as one. A browser password box that says nothing useful in its
    title will not be detected; that limitation is in the write-up rather than
    papered over here.
    """

    def __init__(self, disable_in_password_fields: bool | None = None) -> None:
        self.disable_in_password_fields = (
            disable_in_password_fields
            if disable_in_password_fields is not None
            else get("app.privacy.disable_in_password_fields", True)
        )

    def blocked(self) -> bool:
        if not self.disable_in_password_fields:
            return False
        state = win32.is_password_field()
        if state is True:
            return True
        if state is None:
            return win32.title_suggests_password()
        return False


@dataclass
class HookCallbacks:
    """What the app wants to hear about. Every one is optional."""

    on_update: Callable[[], None] | None = None
    on_accept: Callable[[], None] | None = None
    on_dismiss: Callable[[], None] | None = None
    on_toggle: Callable[[], None] | None = None
    on_next: Callable[[], None] | None = None
    on_previous: Callable[[], None] | None = None

    def dispatch(self, action: Action) -> None:
        handler = {
            Action.UPDATE: self.on_update,
            Action.ACCEPT: self.on_accept,
            Action.DISMISS: self.on_dismiss,
            Action.TOGGLE: self.on_toggle,
            Action.NEXT: self.on_next,
            Action.PREVIOUS: self.on_previous,
        }.get(action)
        if handler is not None:
            handler()


def normalize_key(key) -> KeyEvent:
    """Flatten a ``pynput`` key into a :class:`KeyEvent`.

    pynput hands back either a ``KeyCode`` with a ``char`` or a ``Key`` enum
    member whose name is the thing we care about. Modifier state is tracked by
    the hook, not the key, so it is filled in by the caller.
    """
    char = getattr(key, "char", None)
    if char is not None:
        return KeyEvent(char=char)
    name = getattr(key, "name", None)
    if name is None:
        name = str(key).rsplit(".", 1)[-1]
    return KeyEvent(name=str(name).lower())


class KeystrokeHook:
    """The global keyboard listener.

    Owns nothing but the ``pynput`` listener and the modifier state; every
    decision belongs to the :class:`KeyRouter` it was given. ``start`` is where
    ``pynput`` is imported, so importing this module — which the test suite
    does — never requires the package to be installed.

    A note for the write-up rather than a warning in the log: this is a global
    low-level keyboard hook, which is structurally the same thing a keylogger
    installs, and Windows Defender may say so. Nothing is written to disk and
    nothing leaves the process; the honest mitigation is to disclose it, not to
    hide it.
    """

    MODIFIER_NAMES = {
        "ctrl": "ctrl", "ctrl_l": "ctrl", "ctrl_r": "ctrl",
        "alt": "alt", "alt_l": "alt", "alt_r": "alt", "alt_gr": "alt",
    }

    def __init__(
        self,
        router: KeyRouter | None = None,
        callbacks: HookCallbacks | None = None,
        policy: CapturePolicy | None = None,
        suppress_accept: bool | None = None,
    ) -> None:
        self.router = router if router is not None else KeyRouter()
        self.callbacks = callbacks if callbacks is not None else HookCallbacks()
        self.policy = policy if policy is not None else CapturePolicy()
        self.enabled = True
        self.suppress_accept = bool(
            get("app.hotkeys.suppress_accept", True)
        ) if suppress_accept is None else bool(suppress_accept)
        self._held: set[str] = set()
        self._listener = None

    # ------------------------------------------------------------- lifecycle

    @staticmethod
    def available() -> bool:
        """Whether ``pynput`` can be imported — the editor fallback checks this."""
        try:
            import pynput  # noqa: F401
        except Exception:
            return False
        return True

    def start(self) -> "KeystrokeHook":
        from pynput import keyboard  # imported here: optional dependency

        options = {}
        if self.suppress_accept and self.accept_vk is not None:
            # Ignored by pynput on any backend other than win32, which is the
            # only one that can suppress a single key anyway.
            options["win32_event_filter"] = self.win32_event_filter
        self._listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release, **options
        )
        self._listener.daemon = True
        self._listener.start()
        return self

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self._held.clear()

    @property
    def running(self) -> bool:
        return self._listener is not None

    # ----------------------------------------------------- accept suppression

    @property
    def accept_vk(self) -> int | None:
        """Virtual key code of the configured accept key, or ``None``.

        ``None`` means the key is one this module has no VK for, in which case
        the accept still works -- it just also reaches the application, which
        is the behaviour without the filter.
        """
        return ACCEPT_VK_CODES.get(self.router.accept)

    def win32_event_filter(self, msg, data) -> bool:
        """Swallow the accept key while a suggestion is on screen.

        Without this, accepting a suggestion with Tab types the word *and*
        leaves the Tab itself to reach the application, which indents the line
        or moves focus out of the field. pynput cannot fix that from
        :meth:`on_press`: on Windows the listener callbacks run on a message
        loop that the low-level hook posts to, long after the hook procedure
        has already told Windows to pass the key along. This filter is the one
        callback that runs *inside* the hook procedure, so it is the only place
        a single event can still be stopped -- which means the decision to
        swallow Tab and the accept it stands for both have to be made here.

        Two constraints shape the body. The hook procedure must return quickly,
        because Windows silently unhooks a callback that overruns
        ``LowLevelHooksTimeout`` (~300 ms by default), and the accept types
        text through pynput, which must not run inside the hook it would
        re-enter. So the routing -- which is pure state, and must be
        synchronous, or a second Tab would see ``suggesting`` still true --
        happens here, and only the callback is handed to a worker thread.

        Returning ``True`` leaves the event completely alone: it reaches the
        application and, through the message loop, :meth:`on_press`.
        """
        if msg not in (WM_KEYDOWN, WM_SYSKEYDOWN):
            return True
        if getattr(data, "vkCode", None) != self.accept_vk:
            return True
        # Ctrl+Tab and Alt+Tab belong to the window manager, never to us.
        if self._held or not self.enabled or not self.router.suggesting:
            return True
        if self.policy.blocked():
            return True
        action = self.router.route(KeyEvent(name=self.router.accept))
        if action is Action.ACCEPT:
            # Normally the app clears this when it hides the overlay, but that
            # now happens on the worker thread: a second Tab arriving in the
            # meantime would be suppressed and accepted twice. The suggestion
            # is spoken for the moment it is routed.
            self.router.suggesting = False
        self._dispatch_async(action)
        self._listener.suppress_event()  # raises; nothing below runs
        return False

    def _dispatch_async(self, action: "Action") -> None:
        """Run a callback off the hook thread.

        One thread per accept, which is a keypress the user made deliberately
        -- not per keystroke.
        """
        threading.Thread(
            target=self.callbacks.dispatch, args=(action,), daemon=True
        ).start()

    # --------------------------------------------------------------- events

    def on_press(self, key) -> None:
        name = getattr(key, "name", None)
        modifier = self.MODIFIER_NAMES.get(str(name).lower()) if name else None
        if modifier:
            self._held.add(modifier)
            return
        event = normalize_key(key)
        self.handle(KeyEvent(
            char=event.char,
            name=event.name,
            ctrl="ctrl" in self._held,
            alt="alt" in self._held,
        ))

    def on_release(self, key) -> None:
        name = getattr(key, "name", None)
        modifier = self.MODIFIER_NAMES.get(str(name).lower()) if name else None
        if modifier:
            self._held.discard(modifier)

    def handle(self, event: KeyEvent) -> Action:
        """Route one event and fire the matching callback.

        Split out from :meth:`on_press` so the whole path can be tested with a
        synthetic :class:`KeyEvent` and no listener running.
        """
        if not self.enabled:
            return Action.IGNORE
        if self.policy.blocked():
            # Not merely "do not suggest": drop what was already captured, so
            # a password typed into a field the app then leaves is not sitting
            # in the buffer afterwards.
            self.router.buffer.reset()
            self.callbacks.dispatch(Action.DISMISS)
            return Action.IGNORE
        action = self.router.route(event)
        self.callbacks.dispatch(action)
        return action

    def __repr__(self) -> str:
        state = "running" if self.running else "stopped"
        return f"KeystrokeHook({state}, enabled={self.enabled}, {self.router.buffer!r})"
