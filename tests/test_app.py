"""Phase 7 tests — the desktop layer.

The desktop layer cannot be tested the way the NLP phases were: there is no
held-out set for "did the overlay steal focus". So the module was split by
testability (see :mod:`nticipate.app.hooks`) and this file drives the half that
is pure — the context buffer, the key router, the injection arithmetic, the
overlay's geometry and state machine, and the app's own accept/dismiss wiring
through a detached overlay and a recording injector.

What is *not* covered here, and is verified by hand instead, is the part that
only exists on a screen: whether ``WS_EX_NOACTIVATE`` really prevents focus
theft in Notepad and in a browser. That check is in ``report/notes.md`` as a
manual result, not asserted here, because a passing test that did not actually
look at a window would be worse than an honest gap.
"""

from __future__ import annotations

import time

import pytest

from nticipate.app import win32
from nticipate.app.hooks import (
    DEFAULT_SENTENCE_END,
    Action,
    CapturePolicy,
    ContextBuffer,
    Debouncer,
    HookCallbacks,
    Hotkey,
    KeyEvent,
    KeyRouter,
    KeystrokeHook,
    normalize_key,
    parse_hotkey,
)
from nticipate.app.injector import Injection, Injector, plan_injection
from nticipate.app.overlay import (
    SuggestionOverlay,
    anchor_for,
    clamp_to_screen,
    format_items,
)
from nticipate.ngram import NgramModel
from nticipate.predictor import Mode, Predictor, Suggestion

TRAIN = [
    ["i", "would", "like", "to", "know", "more"],
    ["i", "would", "like", "to", "know", "why"],
    ["i", "would", "like", "to", "thank", "you"],
    ["please", "recommend", "a", "good", "book"],
    ["the", "recent", "report", "was", "good"],
] * 4


def make_predictor(**kw) -> Predictor:
    model = NgramModel(order=3, smoothing="stupid_backoff").fit(TRAIN)
    kw.setdefault("truecase", {"i": "I", "india": "India"})
    return Predictor(model, **kw)


def typed(router: KeyRouter, text: str) -> Action:
    """Feed ``text`` through a router one character at a time."""
    action = Action.IGNORE
    for char in text:
        action = router.route(KeyEvent(char=char))
    return action


# ==========================================================================
# ContextBuffer
# ==========================================================================

def test_buffer_accumulates_typed_characters():
    buffer = ContextBuffer()
    buffer.feed_text("i would like to k")
    assert buffer.text == "i would like to k"


def test_buffer_splits_into_context_and_prefix():
    buffer = ContextBuffer(max_words=2)
    buffer.feed_text("i would like to k")
    assert buffer.words == ["like", "to"]
    assert buffer.prefix == "k"


def test_prefix_is_empty_after_a_space():
    buffer = ContextBuffer(max_words=2)
    buffer.feed_text("i would like to ")
    assert buffer.prefix == ""
    assert buffer.words == ["like", "to"]


def test_buffer_keeps_only_max_context_words():
    # The trigram model can use two words; storing more would be dead weight
    # in the one object that holds what the user typed.
    buffer = ContextBuffer(max_words=2)
    buffer.feed_text("one two three four five ")
    assert buffer.words == ["four", "five"]


@pytest.mark.parametrize("char", list(".!?।॥"))
def test_sentence_end_clears_the_buffer(char):
    # The danda is here because the same buffer serves the Phase 5 Hindi model.
    buffer = ContextBuffer()
    buffer.feed_text("this is a sentence")
    assert buffer.feed(char) is False
    assert buffer.text == ""


def test_default_sentence_end_covers_devanagari():
    assert "।" in DEFAULT_SENTENCE_END and "." in DEFAULT_SENTENCE_END


def test_backspace_removes_characters():
    buffer = ContextBuffer()
    buffer.feed_text("recomm")
    buffer.backspace(2)
    assert buffer.text == "reco"


def test_backspace_past_the_start_empties_rather_than_underflows():
    buffer = ContextBuffer()
    buffer.feed_text("ab")
    buffer.backspace(10)
    assert buffer.text == ""


def test_buffer_is_capped():
    buffer = ContextBuffer(max_chars=20)
    buffer.feed_text("x" * 100)
    assert len(buffer) == 20


def test_buffer_never_offers_a_way_to_persist_itself():
    # Privacy is structural here, not a setting: there is nothing to call that
    # would put the captured text on disk.
    buffer = ContextBuffer()
    for forbidden in ("save", "to_dict", "dump", "write", "path"):
        assert not hasattr(buffer, forbidden)


def test_repr_does_not_leak_what_was_typed():
    buffer = ContextBuffer()
    buffer.feed_text("hunter2 is my password")
    assert "hunter2" not in repr(buffer)
    assert "chars=22" in repr(buffer)


# ==========================================================================
# KeyRouter
# ==========================================================================

def test_typing_asks_for_a_prediction():
    router = KeyRouter()
    assert typed(router, "reco") is Action.UPDATE
    assert router.buffer.prefix == "reco"


def test_enter_resets_the_context_and_dismisses():
    router = KeyRouter()
    typed(router, "some text")
    assert router.route(KeyEvent(name="enter")) is Action.DISMISS
    assert router.buffer.text == ""


def test_return_is_accepted_as_a_spelling_of_enter():
    router = KeyRouter()
    typed(router, "some text")
    router.route(KeyEvent(name="return"))
    assert router.buffer.text == ""


def test_escape_dismisses():
    router = KeyRouter()
    typed(router, "reco")
    assert router.route(KeyEvent(name="escape")) is Action.DISMISS


def test_tab_accepts_while_a_suggestion_is_showing():
    router = KeyRouter()
    typed(router, "reco")
    router.suggesting = True
    assert router.route(KeyEvent(name="tab")) is Action.ACCEPT
    # The buffer is left alone: the app updates it after the injector has
    # actually typed the completion.
    assert router.buffer.text == "reco"


def test_tab_with_nothing_showing_is_an_ordinary_tab():
    # It moves focus to another field, so the sentence is over.
    router = KeyRouter()
    typed(router, "reco")
    router.suggesting = False
    assert router.route(KeyEvent(name="tab")) is Action.DISMISS
    assert router.buffer.text == ""


def test_backspace_updates_and_reprediction_follows():
    router = KeyRouter()
    typed(router, "recon")
    assert router.route(KeyEvent(name="backspace")) is Action.UPDATE
    assert router.buffer.prefix == "reco"


def test_forward_delete_does_not_touch_the_left_context():
    router = KeyRouter()
    typed(router, "reco")
    assert router.route(KeyEvent(name="delete")) is Action.IGNORE
    assert router.buffer.text == "reco"


def test_arrows_move_the_highlight_while_suggesting():
    router = KeyRouter()
    router.suggesting = True
    assert router.route(KeyEvent(name="down")) is Action.NEXT
    assert router.route(KeyEvent(name="up")) is Action.PREVIOUS


def test_arrows_drop_the_context_when_nothing_is_showing():
    # The caret moved somewhere the buffer cannot account for.
    router = KeyRouter()
    typed(router, "reco")
    router.suggesting = False
    assert router.route(KeyEvent(name="left")) is Action.DISMISS
    assert router.buffer.text == ""


def test_a_shortcut_drops_the_context():
    # Ctrl+V changes the document in ways the buffer cannot see.
    router = KeyRouter()
    typed(router, "reco")
    assert router.route(KeyEvent(char="v", ctrl=True)) is Action.DISMISS
    assert router.buffer.text == ""


def test_toggle_hotkey_is_recognised_before_the_shortcut_rule():
    router = KeyRouter(toggle="<ctrl>+<alt>+n")
    assert router.route(KeyEvent(char="n", ctrl=True, alt=True)) is Action.TOGGLE


def test_space_key_name_is_treated_as_a_space():
    router = KeyRouter()
    typed(router, "hello")
    assert router.route(KeyEvent(name="space")) is Action.UPDATE
    assert router.buffer.text == "hello "


def test_unknown_function_keys_are_ignored():
    router = KeyRouter()
    typed(router, "reco")
    assert router.route(KeyEvent(name="f7")) is Action.IGNORE
    assert router.buffer.text == "reco"


# ------------------------------------------------------------------ hotkeys

@pytest.mark.parametrize(
    "spec, expected",
    [
        ("<ctrl>+<alt>+n", Hotkey("n", ctrl=True, alt=True)),
        ("<ctrl>+space", Hotkey("space", ctrl=True)),
        ("ctrl+alt+j", Hotkey("j", ctrl=True, alt=True)),
        ("", None),
        (None, None),
    ],
)
def test_parse_hotkey(spec, expected):
    assert parse_hotkey(spec) == expected


def test_hotkey_requires_exact_modifiers():
    hotkey = Hotkey("n", ctrl=True, alt=True)
    assert hotkey.matches(KeyEvent(char="n", ctrl=True, alt=True))
    assert not hotkey.matches(KeyEvent(char="n", ctrl=True))
    assert not hotkey.matches(KeyEvent(char="n"))


def test_normalize_key_handles_both_pynput_shapes():
    class KeyCode:
        char = "a"

    class NamedKey:
        char = None
        name = "Tab"

    assert normalize_key(KeyCode()) == KeyEvent(char="a")
    assert normalize_key(NamedKey()) == KeyEvent(name="tab")


# ==========================================================================
# Debouncer
# ==========================================================================

def test_debouncer_fires_once_for_a_burst():
    calls: list[int] = []
    debouncer = Debouncer(delay_ms=30)
    for _ in range(5):
        debouncer.schedule(lambda: calls.append(1))
        time.sleep(0.002)
    time.sleep(0.15)
    assert calls == [1]


def test_debouncer_can_be_cancelled():
    calls: list[int] = []
    debouncer = Debouncer(delay_ms=30)
    debouncer.schedule(lambda: calls.append(1))
    debouncer.cancel()
    time.sleep(0.1)
    assert calls == []


def test_debouncer_default_matches_the_configured_budget():
    from nticipate.config import get

    assert Debouncer().delay_ms == get("prediction.latency.debounce_ms")


# ==========================================================================
# CapturePolicy — the privacy rule
# ==========================================================================

def test_capture_is_blocked_in_a_detected_password_field(monkeypatch):
    monkeypatch.setattr(win32, "is_password_field", lambda: True)
    assert CapturePolicy(disable_in_password_fields=True).blocked() is True


def test_capture_is_allowed_in_an_ordinary_field(monkeypatch):
    monkeypatch.setattr(win32, "is_password_field", lambda: False)
    assert CapturePolicy(disable_in_password_fields=True).blocked() is False


def test_unknown_field_falls_back_to_the_window_title(monkeypatch):
    # Browsers and Electron expose nothing about their inputs, so the title is
    # the only signal left. It is a guess, and it is documented as one.
    monkeypatch.setattr(win32, "is_password_field", lambda: None)
    monkeypatch.setattr(win32, "foreground_title", lambda: "Sign in - Gmail")
    assert CapturePolicy(disable_in_password_fields=True).blocked() is True

    monkeypatch.setattr(win32, "foreground_title", lambda: "Untitled - Notepad")
    assert CapturePolicy(disable_in_password_fields=True).blocked() is False


def test_the_policy_can_be_turned_off():
    monkeypatched = CapturePolicy(disable_in_password_fields=False)
    assert monkeypatched.blocked() is False


def test_blocked_capture_discards_what_was_already_typed(monkeypatch):
    # Not merely "stop suggesting": a password typed into a field the app then
    # leaves must not still be sitting in the buffer.
    hook = KeystrokeHook()
    hook.router.buffer.feed_text("secret")
    monkeypatch.setattr(win32, "is_password_field", lambda: True)
    assert hook.handle(KeyEvent(char="x")) is Action.IGNORE
    assert hook.router.buffer.text == ""


def test_title_hints_are_case_insensitive():
    assert win32.title_suggests_password("LOG IN — Bank")
    assert not win32.title_suggests_password("notes.txt - Notepad")


# ==========================================================================
# KeystrokeHook
# ==========================================================================

@pytest.fixture
def open_hook(monkeypatch):
    """A hook whose capture policy always allows, so tests drive the router."""
    monkeypatch.setattr(win32, "is_password_field", lambda: False)
    monkeypatch.setattr(win32, "foreground_title", lambda: "Untitled - Notepad")
    return KeystrokeHook()


def test_hook_dispatches_the_matching_callback(open_hook):
    seen: list[str] = []
    open_hook.callbacks = HookCallbacks(
        on_update=lambda: seen.append("update"),
        on_dismiss=lambda: seen.append("dismiss"),
    )
    open_hook.handle(KeyEvent(char="r"))
    open_hook.handle(KeyEvent(name="enter"))
    assert seen == ["update", "dismiss"]


def test_a_disabled_hook_captures_nothing(open_hook):
    open_hook.enabled = False
    assert open_hook.handle(KeyEvent(char="r")) is Action.IGNORE
    assert open_hook.router.buffer.text == ""


def test_modifier_press_and_release_is_tracked(open_hook):
    class Ctrl:
        char = None
        name = "ctrl_l"

    seen: list[Action] = []
    open_hook.callbacks = HookCallbacks(on_dismiss=lambda: seen.append(Action.DISMISS))
    open_hook.on_press(Ctrl())
    open_hook.on_press(type("K", (), {"char": "c"})())
    assert seen == [Action.DISMISS]        # Ctrl+C is a shortcut, not typing
    open_hook.on_release(Ctrl())
    open_hook.on_press(type("K", (), {"char": "c"})())
    assert open_hook.router.buffer.text == "c"


def test_hook_reports_pynput_availability():
    assert KeystrokeHook.available() in (True, False)


# ==========================================================================
# Injector — the arithmetic
# ==========================================================================

def test_completion_types_only_the_missing_tail():
    assert plan_injection("rec", "recommend") == Injection(0, "ommend ")


def test_next_word_types_the_whole_word():
    assert plan_injection("", "know") == Injection(0, "know ")


def test_casing_mismatch_retypes_the_word():
    # The predictor truecases its output. Appending "ia" to "ind" would leave
    # "indIa" on screen, so the typed prefix is deleted first.
    assert plan_injection("ind", "India") == Injection(3, "India ")


def test_trailing_space_can_be_turned_off():
    assert plan_injection("rec", "recommend", append_space=False) == Injection(0, "ommend")


def test_accepting_the_word_already_typed_is_a_no_op_plus_a_space():
    assert plan_injection("the", "the") == Injection(0, " ")


def test_keystroke_count_includes_the_backspaces():
    assert plan_injection("ind", "India").keystrokes == 3 + len("India ")


class RecordingInjector(Injector):
    """An injector that plans normally and records instead of typing."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.sent: list[Injection] = []

    def send(self, injection: Injection) -> None:
        self.sent.append(injection)
        self.injected_keystrokes += injection.keystrokes
        self.injections += 1


def test_injector_accept_plans_and_sends():
    injector = RecordingInjector()
    injection = injector.accept("rec", "recommend")
    assert injector.sent == [injection] == [Injection(0, "ommend ")]
    assert injector.injections == 1


def test_empty_injection_is_not_sent():
    injector = Injector()
    injector.send(Injection(0, ""))     # must not import pynput at all
    assert injector.injections == 0


# ==========================================================================
# Overlay — geometry and state, without a screen
# ==========================================================================

def test_format_items_numbers_the_suggestions():
    items = [Suggestion("know", -1.0, "base", Mode.NEXT_WORD),
             Suggestion("keep", -2.0, "base", Mode.NEXT_WORD)]
    assert format_items(items, 3) == ["1. know", "2. keep"]


def test_format_items_respects_max_items():
    assert format_items(["a", "b", "c", "d"], 2) == ["1. a", "2. b"]


def test_anchor_prefers_the_caret():
    caret = win32.Rect(100, 200, 102, 216)
    point = anchor_for(caret, win32.Point(900, 900), offset_x=4, offset_y=20)
    assert point == win32.Point(104, 236)


def test_anchor_falls_back_to_the_mouse():
    # This is the browser and Electron case, and it is the common one.
    point = anchor_for(None, win32.Point(900, 900), offset_x=4, offset_y=20)
    assert point == win32.Point(904, 920)


def test_anchor_gives_up_when_neither_is_available():
    # Better no popup than a popup in an arbitrary corner.
    assert anchor_for(None, None) is None


def test_clamp_pulls_a_window_back_from_the_right_edge():
    point = clamp_to_screen(1500, 100, 200, 80, 1536, 864)
    assert point.x + 200 <= 1536


def test_clamp_flips_above_the_anchor_at_the_bottom_edge():
    # Sliding up would put the list on top of the text being typed.
    point = clamp_to_screen(100, 850, 200, 80, 1536, 864)
    assert point.y < 850
    assert point.y + 80 <= 864


def test_clamp_is_a_no_op_when_the_screen_size_is_unknown():
    assert clamp_to_screen(10, 10, 100, 50, 0, 0) == win32.Point(10, 10)


def test_detached_overlay_tracks_state_without_drawing():
    overlay = SuggestionOverlay()
    assert overlay.attached is False
    overlay.show(["know", "keep", "knew"])
    assert overlay.visible and overlay.current() == "know"


def test_overlay_navigation_wraps():
    overlay = SuggestionOverlay(max_items=3)
    overlay.show(["know", "keep", "knew"])
    overlay.move(1)
    assert overlay.current() == "keep"
    overlay.move(-2)
    assert overlay.current() == "knew"


def test_overlay_hide_resets_the_selection():
    overlay = SuggestionOverlay()
    overlay.show(["a", "b"])
    overlay.move(1)
    overlay.hide()
    assert overlay.current() is None
    assert overlay.selected == 0


def test_showing_nothing_hides_the_overlay():
    overlay = SuggestionOverlay()
    overlay.show(["a"])
    overlay.show([])
    assert overlay.visible is False


def test_overlay_honours_max_items():
    overlay = SuggestionOverlay(max_items=2)
    overlay.show(["a", "b", "c"])
    overlay.move(1)
    overlay.move(1)
    assert overlay.current() == "a"          # wrapped after two, not three


# ==========================================================================
# NticipateApp — the wiring
# ==========================================================================

@pytest.fixture
def app(monkeypatch):
    """The real app, with the trained-on-nothing predictor and no I/O."""
    from nticipate.app.tray import NticipateApp

    monkeypatch.setattr(win32, "is_password_field", lambda: False)
    instance = NticipateApp(
        predictor=make_predictor(),
        overlay=SuggestionOverlay(),
        injector=RecordingInjector(),
    )
    yield instance
    instance.debouncer.cancel()


def test_app_predicts_from_typed_keys(app):
    for char in "i would like to k":
        app.hook.handle(KeyEvent(char=char))
    app.predict_now()
    assert [s.word for s in app.suggestions][:1] == ["know"]
    assert app.overlay.current() == "know"
    assert app.router.suggesting is True


def test_accept_injects_the_tail_and_advances_the_buffer(app):
    for char in "i would like to k":
        app.hook.handle(KeyEvent(char=char))
    app.predict_now()
    app.hook.handle(KeyEvent(name="tab"))
    assert app.injector.sent == [Injection(0, "now ")]
    # The buffer now reflects what is on screen, so the next prediction is a
    # next-word prediction rather than a completion of a finished word.
    assert app.buffer.text == "i would like to know "
    assert app.overlay.visible is False


def test_accept_uses_the_highlighted_suggestion_not_the_first(app):
    for char in "i would like to ":
        app.hook.handle(KeyEvent(char=char))
    app.predict_now()
    second = app.suggestions[1].word
    app.hook.handle(KeyEvent(name="down"))
    app.hook.handle(KeyEvent(name="tab"))
    assert app.injector.sent[0].text.startswith(second)


def test_sentence_end_clears_everything(app):
    for char in "i would like to know":
        app.hook.handle(KeyEvent(char=char))
    app.predict_now()
    app.hook.handle(KeyEvent(char="."))
    assert app.buffer.text == ""
    assert app.overlay.visible is False


def test_toggle_disables_capture_and_clears_the_buffer(app):
    for char in "i would":
        app.hook.handle(KeyEvent(char=char))
    app.toggle_enabled()
    assert app.enabled is False and app.hook.enabled is False
    assert app.buffer.text == ""
    app.hook.handle(KeyEvent(char="x"))
    assert app.buffer.text == ""


def test_empty_buffer_predicts_nothing(app):
    app.predict_now()
    assert app.suggestions == []
    assert app.overlay.visible is False


def test_a_prediction_failure_does_not_kill_the_hook(app, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("model exploded")

    app.buffer.feed_text("i would ")
    monkeypatch.setattr(app.predictor, "suggest", boom)
    app.predict_now()                     # must not raise
    assert app.overlay.visible is False
    # And the app still works once the model does.
    monkeypatch.undo()
    app.predict_now()
    assert app.suggestions


def test_learning_is_off_until_asked_for(app):
    assert app.learning is False
    before = app.predictor.profile.token_count
    app.buffer.feed_text("i would like to k")
    app.predict_now()
    app.hook.handle(KeyEvent(name="tab"))
    assert app.predictor.profile.token_count == before


def test_learning_when_enabled_reaches_the_profile(app):
    app.toggle_learning()
    assert app.learning is True
    before = app.predictor.profile.token_count
    app.buffer.feed_text("i would like to k")
    app.predict_now()
    app.hook.handle(KeyEvent(name="tab"))
    assert app.predictor.profile.token_count > before


def test_status_reports_timings_and_no_text(app):
    app.buffer.feed_text("i would ")
    app.predict_now()
    status = app.status()
    assert status["predictions"] == 1
    assert status["p95_ms"] >= 0.0
    assert "would" not in repr(status)


def test_warmup_drops_its_own_timing(app):
    elapsed = app.warmup()
    assert elapsed >= 0.0
    assert app.status()["predictions"] == 0


def test_prediction_stays_inside_the_debounce_budget(app):
    from nticipate.config import get

    app.warmup()
    for text in ("i would ", "i would like to ", "reco", "i would like to k"):
        app.buffer.reset()
        app.buffer.feed_text(text)
        app.predict_now()
    budget = get("prediction.latency.p95_budget_ms")
    assert max(app.latencies) < budget, f"slowest prediction {max(app.latencies):.1f} ms"


def test_app_repr_does_not_leak_the_buffer(app):
    app.buffer.feed_text("my secret sentence")
    assert "secret" not in repr(app)


# ==========================================================================
# Editor fallback
# ==========================================================================

def test_editor_finds_the_current_sentence():
    from nticipate.app.editor import current_sentence

    assert current_sentence("Done. I would like to k") == " I would like to k"


def test_editor_sentence_is_the_whole_text_when_none_has_ended():
    from nticipate.app.editor import current_sentence

    assert current_sentence("I would like") == "I would like"


def test_editor_splits_on_the_danda_too():
    from nticipate.app.editor import current_sentence

    assert current_sentence("यह वाक्य है। अब") == " अब"


# ==========================================================================
# Win32 layer — behaviour that must hold on every platform
# ==========================================================================

def test_win32_helpers_never_raise_off_a_real_window():
    # Every one of these legitimately fails in a browser or on CI; the
    # contract is that failure is a return value, not an exception.
    for call in (win32.caret_rect, win32.mouse_position, win32.is_password_field,
                 win32.foreground_title, win32.screen_size, win32.get_clipboard_text):
        call()


def test_set_no_activate_rejects_a_null_handle():
    assert win32.set_no_activate(None) is False


def test_opaque_classes_include_the_browsers_and_electron():
    # The caret query returns a meaningless rectangle for these, so they are
    # answered with "unknown" and fall through to the mouse anchor.
    assert "Chrome_RenderWidgetHostHWND" in win32.OPAQUE_WINDOW_CLASSES
    assert "MozillaWindowClass" in win32.OPAQUE_WINDOW_CLASSES


# ==========================================================================
# Config contract
# ==========================================================================

def test_injection_method_is_one_of_the_two_implemented():
    from nticipate.config import require

    assert require("app.capture.injection_method") in {"type", "clipboard"}


def test_overlay_theme_exists():
    from nticipate.app.overlay import THEMES
    from nticipate.config import require

    assert require("app.overlay.theme") in THEMES


def test_capture_window_matches_the_model_order():
    # Raising app.capture.max_context_words above order-1 stores text the
    # model cannot use — which is exactly what the privacy rule forbids.
    from nticipate.config import require

    assert require("app.capture.max_context_words") <= require("ngram.max_order") - 1


def test_learning_defaults_to_off():
    # It is the one feature that writes what the user typed to disk, so it is
    # opt-in from the tray menu rather than a default.
    from nticipate.config import require

    assert require("app.learning.enabled") is False
