"""Phase 7 — the fallback: a plain editor window with live suggestions.

The plan names this as the contingency if the global hook proves unworkable,
and it earns its place for two reasons beyond that. It is the demo that always
works — no Defender warning, no caret guessing, no dependency on ``pynput`` —
and it is the only way to watch the predictor react to typing without a global
hook running at all.

It reuses the same objects as the tray app: the same
:class:`~nticipate.predictor.Predictor`, the same ranking, the same
:func:`~nticipate.app.injector.plan_injection` arithmetic on Tab. What it does
*not* use is the capture layer, because a Tk ``Text`` widget already knows what
is in it — there is no buffer to reconstruct and nothing to guess about the
caret. That makes it a weaker product and an equally valid demonstration of
every NLP component, which is the trade the plan describes.
"""

from __future__ import annotations

import logging
import time

from nticipate.config import get
from nticipate.predictor import Predictor
from nticipate.app.hooks import DEFAULT_SENTENCE_END
from nticipate.app.injector import plan_injection

log = logging.getLogger("nticipate.app.editor")

WINDOW_TITLE = "N-Ticipate — editor"


def current_sentence(text: str, sentence_end: str = DEFAULT_SENTENCE_END) -> str:
    """The text since the last sentence boundary.

    The equivalent of :class:`~nticipate.app.hooks.ContextBuffer`'s job, done
    by scanning instead of accumulating — here the whole document is available,
    so there is no reason to keep a shadow copy of it.
    """
    cut = max((text.rfind(char) for char in sentence_end), default=-1)
    return text[cut + 1:] if cut >= 0 else text


class EditorApp:
    """A Tk text editor that suggests as you type."""

    def __init__(self, predictor: Predictor | None = None, learning: bool | None = None) -> None:
        if predictor is None:
            from nticipate.app.tray import load_predictor

            predictor = load_predictor()
        self.predictor = predictor
        self.learning = learning if learning is not None else bool(
            get("app.learning.enabled", False)
        )
        self.max_items = get("app.overlay.max_items", 3)
        self.debounce_ms = get("prediction.latency.debounce_ms", 50)
        self.suggestions: list = []
        self.selected = 0
        self.latencies: list[float] = []
        self._pending = None
        self._root = None
        self._text = None
        self._list = None
        self._status = None

    # ------------------------------------------------------------------ UI

    def build(self):
        import tkinter as tk

        root = tk.Tk()
        root.title(WINDOW_TITLE)
        root.geometry("760x460")

        text = tk.Text(root, wrap="word", font=("Consolas", 12), undo=True,
                       padx=10, pady=10)
        text.pack(fill="both", expand=True)
        text.focus_set()

        suggestions = tk.Label(root, text="", anchor="w", justify="left",
                               font=("Segoe UI", get("app.overlay.font_size", 11)),
                               background="#1e1e1e", foreground="#d4d4d4", padx=10, pady=4)
        suggestions.pack(fill="x")

        status = tk.Label(root, text="", anchor="w", font=("Segoe UI", 9),
                          foreground="#666666", padx=10)
        status.pack(fill="x")

        # Tab accepts, so it must not insert a tab character; returning "break"
        # is Tk's way of saying the default binding is cancelled.
        text.bind("<Tab>", self._on_accept)
        text.bind("<Escape>", self._on_dismiss)
        text.bind("<Down>", self._on_next)
        text.bind("<Up>", self._on_previous)
        text.bind("<KeyRelease>", self._on_key)

        self._root, self._text, self._list, self._status = root, text, suggestions, status
        self.warmup()
        self._render()
        return root

    def warmup(self) -> float:
        """Pay the first-prediction cost before the window is usable.

        Same reasoning as :meth:`~nticipate.app.tray.NticipateApp.warmup`: the
        first call loads NLTK's tokenizer and builds the model's lazy index,
        and is ~500 ms against ~1 ms afterwards.
        """
        start = time.perf_counter()
        self.predictor.suggest("the ")
        self.suggestions = []
        return (time.perf_counter() - start) * 1000.0

    def run(self) -> None:
        self.build().mainloop()
        if self.learning and self.predictor.profile:
            self.predictor.profile.save()

    # ------------------------------------------------------------- events

    def _on_key(self, event=None):
        # Keys that are handled by their own binding must not also re-predict.
        if event is not None and event.keysym in {"Tab", "Escape", "Up", "Down"}:
            return None
        if self._pending is not None:
            self._root.after_cancel(self._pending)
        self._pending = self._root.after(self.debounce_ms, self.predict_now)
        return None

    def predict_now(self) -> None:
        self._pending = None
        buffer = current_sentence(self._text.get("1.0", "insert"))
        if not buffer.strip():
            self.suggestions = []
            self._render()
            return
        start = time.perf_counter()
        self.suggestions = self.predictor.suggest(buffer, k=self.max_items)
        self.latencies.append((time.perf_counter() - start) * 1000.0)
        self.selected = 0
        self._render()

    def _on_accept(self, event=None):
        if not self.suggestions:
            return None
        word = self.suggestions[self.selected].word
        buffer = current_sentence(self._text.get("1.0", "insert"))
        _, prefix = self.predictor.split_buffer(buffer)
        injection = plan_injection(prefix, word, get("app.capture.append_space", True))
        if injection.backspaces:
            self._text.delete(f"insert-{injection.backspaces}c", "insert")
        self._text.insert("insert", injection.text)
        if self.learning:
            self.predictor.learn(word)
        self.suggestions = []
        self._render()
        return "break"          # swallow the Tab so it does not indent

    def _on_dismiss(self, event=None):
        self.suggestions = []
        self._render()
        return "break"

    def _on_next(self, event=None):
        return self._move(1)

    def _on_previous(self, event=None):
        return self._move(-1)

    def _move(self, delta: int):
        if not self.suggestions:
            return None                     # let the arrow move the caret
        self.selected = (self.selected + delta) % len(self.suggestions)
        self._render()
        return "break"

    # ----------------------------------------------------------- rendering

    def _render(self) -> None:
        if self._list is None:
            return
        if not self.suggestions:
            self._list.configure(text="type to see suggestions   ·   Tab accepts   "
                                      "·   ↑↓ chooses   ·   Esc dismisses")
        else:
            parts = []
            for index, item in enumerate(self.suggestions):
                marker = "▸" if index == self.selected else " "
                tag = f" [{item.tag}]" if item.tag else ""
                parts.append(f"{marker} {index + 1}. {item.word}{tag}")
            self._list.configure(text="   ".join(parts))
        if self._status is not None:
            self._status.configure(text=self._status_line())

    def _status_line(self) -> str:
        if not self.latencies:
            return f"{self.predictor!r}"
        recent = self.latencies[-100:]
        mean = sum(recent) / len(recent)
        worst = max(recent)
        rerank = "POS reranking on" if self.predictor.rerank_active else "no reranking"
        learning = "learning on" if self.learning else "learning off"
        return (f"{len(self.latencies)} predictions · mean {mean:.1f} ms · "
                f"max {worst:.1f} ms · {rerank} · {learning}")


def run(learning: bool | None = None) -> int:
    """Entry point for ``python -m nticipate.app --editor``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    EditorApp(learning=learning).run()
    return 0
