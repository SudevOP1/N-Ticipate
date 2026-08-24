"""Phase 7 — the tray icon and the object that wires the whole app together.

:class:`NticipateApp` is the only place where the six moving parts meet: the
:class:`~nticipate.predictor.Predictor` from Phases 3 and 6, the
:class:`~nticipate.app.hooks.ContextBuffer` and hook, the
:class:`~nticipate.app.hooks.Debouncer`, the
:class:`~nticipate.app.overlay.SuggestionOverlay` and the
:class:`~nticipate.app.injector.Injector`. Everything it does is a method that
can be called directly, so the app can be driven from a test or from the
editor fallback without a tray, a hook or a screen.

:class:`TrayIcon` is the pystray adapter and holds no state of its own.

**Threads.** Three of them, and the split is forced rather than chosen. pynput
delivers keystrokes on its own listener thread; Tk demands that its widgets are
only touched from the thread that created the root; pystray wants a run loop.
So: the hook thread mutates the buffer and schedules work, the debouncer's
timer thread runs the prediction (which is pure computation and touches no
widget), and the overlay marshals every draw back to the Tk thread with
``after``. The main thread runs Tk and the tray.

**Learning is opt-in.** The Phase 3 personalisation layer improves with use,
but it improves by *storing what the user wrote*, and Phase 7's privacy
requirement is that the context buffer never reaches disk. Both are satisfiable
at once, but not silently: learning is off until the user turns on "Learn from
my typing" in the tray menu. What is then saved is the profile's n-gram counts
(``data/models/user_profile.json``), never the keystroke stream, and never
anything typed while capture was blocked.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from nticipate.config import get, reload_config, resolve_path
from nticipate.predictor import Predictor, Suggestion
from nticipate.app import win32
from nticipate.app.hooks import (
    CapturePolicy,
    ContextBuffer,
    Debouncer,
    HookCallbacks,
    KeyRouter,
    KeystrokeHook,
)
from nticipate.app.injector import Injector
from nticipate.app.overlay import SuggestionOverlay

log = logging.getLogger("nticipate.app")

#: How many prediction timings to keep for the tray's status line. Enough to
#: make the p95 mean something, small enough to reflect the current session
#: rather than an average over the whole day.
LATENCY_WINDOW = 200

#: Throwaway context for :meth:`NticipateApp.warmup`. A single common word is
#: enough to touch every lazy path: the tokenizer, the model's context index,
#: the trie and the tagger.
WARMUP_TEXT = "the "


def model_paths() -> dict[str, Path | None]:
    """Resolve the artefacts the app loads at start-up.

    A missing tagger is not an error: without it the predictor is exactly
    Phase 3, which is a working app. A missing n-gram model is fatal, because
    there is nothing to suggest.
    """
    def maybe(key: str, default: str) -> Path | None:
        value = get(key, default)
        if not value:
            return None
        path = resolve_path(value)
        return path if path.exists() else None

    return {
        "ngram": maybe("app.models.ngram", "data/models/ngram_trigram_pruned.pkl"),
        "corpus": maybe("app.models.corpus", "data/processed/brown.json"),
        "truecase": maybe("app.models.truecase", "data/models/truecase.json"),
        "tagger": maybe("app.models.tagger", "data/models/hmm_english.pkl"),
        "tagger_hindi": maybe("app.models.tagger_hindi", "data/models/hmm_hindi.pkl"),
        "profile": resolve_path(get("paths.user_profile", "data/models/user_profile.json")),
    }


def load_predictor(language: str | None = None) -> Predictor:
    """Build the predictor the app runs on, from the packaged artefacts."""
    paths = model_paths()
    if paths["ngram"] is None:
        raise FileNotFoundError(
            "No packaged n-gram model. Run notebooks/02_ngram_models.ipynb to "
            "build data/models/ngram_trigram_pruned.pkl, or point "
            "app.models.ngram at one."
        )
    language = language or get("app.language", "english")
    tagger_key = "tagger_hindi" if language == "hindi" else "tagger"
    tagger_path = paths[tagger_key]
    if tagger_path is None and get("reranking.enabled", True):
        log.warning(
            "No %s tagger found; running without POS reranking (Phase 3 behaviour).",
            language,
        )
    profile_path = paths["profile"] if paths["profile"].exists() else None
    # The truecase map is all the corpus is ever read for. Prefer the small
    # artefact; fall back to the full corpus only when it has not been built.
    truecase_path = paths["truecase"]
    corpus_path = None if truecase_path is not None else paths["corpus"]
    if truecase_path is None and corpus_path is not None:
        log.warning(
            "No truecase artefact; loading %s for its map alone. Run "
            "scripts/retrain.py (or Corpus.save_truecase) to build "
            "data/models/truecase.json.",
            corpus_path.name,
        )
    return Predictor.from_paths(
        paths["ngram"],
        corpus_path=corpus_path,
        profile_path=profile_path,
        tagger_path=tagger_path,
        truecase_path=truecase_path,
    )


class NticipateApp:
    """The running application: capture -> predict -> display -> inject."""

    def __init__(
        self,
        predictor: Predictor | None = None,
        overlay: SuggestionOverlay | None = None,
        injector: Injector | None = None,
        hook: KeystrokeHook | None = None,
        language: str | None = None,
        learning: bool | None = None,
    ) -> None:
        self.language = language or get("app.language", "english")
        self.predictor = predictor if predictor is not None else load_predictor(self.language)
        self.overlay = overlay if overlay is not None else SuggestionOverlay()
        self.injector = injector if injector is not None else Injector()

        self.buffer = ContextBuffer()
        self.router = KeyRouter(buffer=self.buffer)
        self.debouncer = Debouncer()
        self.hook = hook if hook is not None else KeystrokeHook(
            router=self.router,
            callbacks=HookCallbacks(
                on_update=self.on_update,
                on_accept=self.on_accept,
                on_dismiss=self.on_dismiss,
                on_toggle=self.toggle_enabled,
                on_next=lambda: self.overlay.move(1),
                on_previous=lambda: self.overlay.move(-1),
            ),
            policy=CapturePolicy(),
        )

        self.enabled = bool(get("app.enabled_on_start", True))
        self.hook.enabled = self.enabled
        self.learning = learning if learning is not None else bool(
            get("app.learning.enabled", False)
        )
        self.autosave_every = get("app.learning.autosave_every", 200)
        self._unsaved_tokens = 0

        #: Prediction latencies in milliseconds, for the tray status line and
        #: the Phase 8 table. Timings only -- no text.
        self.latencies: deque[float] = deque(maxlen=LATENCY_WINDOW)
        self.suggestions: list[Suggestion] = []
        self._root = None
        self._icon = None
        self._stopping = threading.Event()

    # -------------------------------------------------------------- pipeline

    def on_update(self) -> None:
        """A key changed the buffer: schedule a prediction after the pause."""
        if not self.enabled:
            return
        self.debouncer.schedule(self.predict_now)

    def predict_now(self) -> None:
        """Run the predictor against the current buffer and draw the result.

        Runs on the debouncer's timer thread. Nothing here touches a widget:
        :meth:`SuggestionOverlay.show` marshals onto the Tk thread itself.
        """
        text = self.buffer.text
        if not text:
            self.hide()
            return
        start = time.perf_counter()
        try:
            suggestions = self.predictor.suggest(text)
        except Exception:
            # A prediction failure must never kill the hook thread and leave
            # the app silently capturing keys with nothing to show for it.
            log.exception("prediction failed")
            self.hide()
            return
        self.latencies.append((time.perf_counter() - start) * 1000.0)
        self.suggestions = suggestions
        if not suggestions:
            self.hide()
            return
        self.router.suggesting = True
        self.overlay.show(suggestions)

    def on_accept(self) -> None:
        """Tab with the overlay up: type the highlighted suggestion."""
        word = self.overlay.current() or (
            self.suggestions[0].word if self.suggestions else None
        )
        if not word:
            return
        _, prefix = self.predictor.split_buffer(self.buffer.text)
        injection = self.injector.accept(prefix, word)
        # Keep the buffer in step with what is now on screen, so the next
        # prediction sees the accepted word as context rather than re-offering
        # a completion of a word that is already finished.
        self.buffer.backspace(injection.backspaces)
        self.buffer.feed_text(injection.text)
        self.hide()
        self._learn(word)

    def on_dismiss(self) -> None:
        self.hide()

    def hide(self) -> None:
        self.router.suggesting = False
        self.overlay.hide()

    # -------------------------------------------------------------- learning

    def _learn(self, text: str) -> None:
        """Feed accepted text to the user profile, if the user allowed it."""
        if not self.learning or not self.predictor.profile:
            return
        self._unsaved_tokens += self.predictor.learn(text)
        if self.autosave_every and self._unsaved_tokens >= self.autosave_every:
            self.save_profile()

    def save_profile(self) -> Path | None:
        if not self.predictor.profile:
            return None
        path = self.predictor.profile.save(
            resolve_path(get("paths.user_profile", "data/models/user_profile.json"))
        )
        self._unsaved_tokens = 0
        log.info("user profile saved (%s)", path)
        return path

    def reset_profile(self) -> None:
        if self.predictor.profile:
            self.predictor.profile.reset()
            self._unsaved_tokens = 0
            log.info("user profile cleared")

    # ---------------------------------------------------------------- state

    def toggle_enabled(self, *_) -> bool:
        self.enabled = not self.enabled
        self.hook.enabled = self.enabled
        if not self.enabled:
            self.buffer.reset()
            self.hide()
        log.info("suggestions %s", "enabled" if self.enabled else "disabled")
        self._refresh_menu()
        return self.enabled

    def toggle_learning(self, *_) -> bool:
        self.learning = not self.learning
        if not self.learning:
            self.save_profile()
        log.info("learning %s", "on" if self.learning else "off")
        self._refresh_menu()
        return self.learning

    def set_language(self, language: str) -> bool:
        """Swap the tagger between the Phase 4 and Phase 5 models.

        Only the tagger changes: the n-gram model is English-trained, so this
        is honestly a POS-reranking language switch rather than a full language
        switch, and it is labelled that way in the menu.
        """
        paths = model_paths()
        key = "tagger_hindi" if language == "hindi" else "tagger"
        path = paths[key]
        if path is None:
            log.warning("no %s tagger available; language unchanged", language)
            return False
        from nticipate.hmm import HMMTagger

        self.predictor.attach_tagger(HMMTagger.load(path))
        self.language = language
        log.info("tagger language set to %s", language)
        self._refresh_menu()
        return True

    def reload_settings(self) -> None:
        """Re-read config.yaml without restarting — the tray's Settings action."""
        reload_config()
        self.buffer = ContextBuffer()
        self.router = KeyRouter(buffer=self.buffer)
        self.hook.router = self.router
        self.debouncer = Debouncer()
        self.overlay.max_items = get("app.overlay.max_items", 3)
        self.predictor.rerank_alpha = get("reranking.alpha", 0.1)
        self.predictor.max_suggestions = get("prediction.max_suggestions", 3)
        log.info("settings reloaded")

    def status(self) -> dict:
        """What the tray shows and what Phase 8 records. No captured text."""
        timings = sorted(self.latencies)
        percentile = lambda p: (  # noqa: E731 - one-liner, used twice
            timings[min(len(timings) - 1, int(p / 100 * len(timings)))] if timings else 0.0
        )
        return {
            "enabled": self.enabled,
            "learning": self.learning,
            "language": self.language,
            "reranking": self.predictor.rerank_active,
            "predictions": len(self.latencies),
            "p50_ms": round(percentile(50), 2),
            "p95_ms": round(percentile(95), 2),
            "injections": self.injector.injections,
        }

    # -------------------------------------------------------------- lifecycle

    def warmup(self) -> float:
        """Run one throwaway prediction, and return how long it took in ms.

        The first prediction is two orders of magnitude slower than the rest —
        it pays for NLTK's tokenizer load and the n-gram model's lazy context
        index. Measured on the packaged trigram model: ~500 ms for the first
        call against ~1.2 ms steady state. Paying it here means the debounce
        budget is only ever measured against the warm path, and the user's
        first keystroke does not visibly stall.

        The timings it produces are dropped, for the same reason: a p95 that
        included the warm-up would describe start-up, not typing.
        """
        start = time.perf_counter()
        try:
            self.predictor.suggest(WARMUP_TEXT)
        except Exception:
            log.exception("warm-up prediction failed")
        elapsed = (time.perf_counter() - start) * 1000.0
        self.latencies.clear()
        self.suggestions = []
        log.info("warm-up prediction: %.0f ms", elapsed)
        return elapsed

    def start(self) -> "NticipateApp":
        """Start capture. Requires a Tk root to already be attached."""
        if not KeystrokeHook.available():
            raise RuntimeError(
                "pynput is not installed — the global hook cannot start. "
                "Run the editor fallback instead: python -m nticipate.app --editor"
            )
        self.warmup()
        self.hook.start()
        log.info("keystroke hook running (%s)", self.predictor)
        return self

    def stop(self) -> None:
        self._stopping.set()
        self.debouncer.cancel()
        self.hook.stop()
        self.buffer.reset()
        if self.learning:
            self.save_profile()
        self.overlay.destroy()
        if self._icon is not None:
            self._icon.stop()
        if self._root is not None:
            try:
                self._root.quit()
            except Exception:
                pass

    def run(self) -> None:
        """Run the tray app: Tk on the main thread, tray and hook alongside."""
        import tkinter as tk

        self._root = tk.Tk()
        self._root.withdraw()          # the app has no main window, only popups
        self.overlay.attach(self._root)
        self.start()

        self._icon = TrayIcon(self)
        self._icon.start()             # pystray runs detached; Tk owns the main thread
        try:
            self._root.mainloop()
        finally:
            self.stop()

    def _refresh_menu(self) -> None:
        if self._icon is not None:
            self._icon.refresh()

    def __repr__(self) -> str:
        return (
            f"NticipateApp(enabled={self.enabled}, language={self.language!r}, "
            f"learning={self.learning}, {self.predictor!r})"
        )


class TrayIcon:
    """The pystray menu. Every item is a method call on :class:`NticipateApp`."""

    def __init__(self, app: NticipateApp) -> None:
        self.app = app
        self._icon = None

    @staticmethod
    def available() -> bool:
        try:
            import pystray  # noqa: F401
            from PIL import Image  # noqa: F401
        except Exception:
            return False
        return True

    @staticmethod
    def make_image(size: int = 64):
        """Draw the tray icon: an 'N' on a dark rounded square.

        Generated rather than shipped as a .ico so PyInstaller in Phase 8 has
        one less data file to bundle.
        """
        from PIL import Image, ImageDraw

        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=size // 6,
                               fill=(30, 30, 30, 255))
        bar, inset = size // 8, size // 4
        top, bottom = inset, size - inset
        draw.rectangle([inset, top, inset + bar, bottom], fill=(220, 220, 220, 255))
        draw.rectangle([size - inset - bar, top, size - inset, bottom],
                       fill=(220, 220, 220, 255))
        draw.line([inset, top, size - inset, bottom], fill=(220, 220, 220, 255),
                  width=bar)
        return image

    def _menu(self):
        import pystray

        item = pystray.MenuItem
        app = self.app
        return pystray.Menu(
            item(lambda _: f"N-Ticipate — {'on' if app.enabled else 'off'}", None,
                 enabled=False),
            pystray.Menu.SEPARATOR,
            item("Suggestions enabled", lambda: app.toggle_enabled(),
                 checked=lambda _: app.enabled),
            item("Learn from my typing", lambda: app.toggle_learning(),
                 checked=lambda _: app.learning),
            pystray.Menu.SEPARATOR,
            item("Tagger language", pystray.Menu(
                item("English", lambda: app.set_language("english"),
                     checked=lambda _: app.language == "english", radio=True),
                item("Hindi", lambda: app.set_language("hindi"),
                     checked=lambda _: app.language == "hindi", radio=True),
            )),
            item("Save profile now", lambda: app.save_profile(),
                 enabled=lambda _: app.predictor.profile is not None),
            item("Forget what I have typed", lambda: app.reset_profile(),
                 enabled=lambda _: app.predictor.profile is not None),
            pystray.Menu.SEPARATOR,
            item("Open settings (config.yaml)", lambda: self.open_config()),
            item("Reload settings", lambda: app.reload_settings()),
            item(lambda _: self._status_line(), None, enabled=False),
            pystray.Menu.SEPARATOR,
            item("Quit", lambda: app.stop()),
        )

    def _status_line(self) -> str:
        status = self.app.status()
        return (
            f"p50 {status['p50_ms']:.0f} ms · p95 {status['p95_ms']:.0f} ms · "
            f"{status['predictions']} predictions"
        )

    @staticmethod
    def open_config() -> None:
        from nticipate.config import config_path

        path = str(config_path())
        if sys.platform == "win32":
            os.startfile(path)  # noqa: S606 - opening the user's own config
        else:
            subprocess.Popen(["xdg-open", path])

    def start(self) -> None:
        import pystray

        self._icon = pystray.Icon(
            "nticipate", self.make_image(), "N-Ticipate", self._menu()
        )
        self.app._icon = self
        self._icon.run_detached()

    def refresh(self) -> None:
        if self._icon is not None:
            self._icon.update_menu()

    def stop(self) -> None:
        if self._icon is not None:
            self._icon.stop()
            self._icon = None


def run(learning: bool | None = None) -> int:
    """Entry point for ``python -m nticipate.app``."""
    logging.basicConfig(
        level=getattr(logging, str(get("logging.level", "INFO")).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not win32.IS_WINDOWS:
        log.warning("N-Ticipate's hook and overlay are Windows-only; "
                    "the editor fallback works everywhere.")
    NticipateApp(learning=learning).run()
    return 0
