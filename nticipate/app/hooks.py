"""
Phase 7: global keystroke capture (pynput) + rolling context buffer.

Buffer resets on Enter / mouse click / window focus change.
Debounced at CFG['app']['debounce_ms'] so fast typists don't trigger
a model call per character.

Password-field detection (CFG['app']['disable_on_password_fields']) must
short-circuit capture entirely -- check this before touching the buffer.
"""

from nticipate.config import load_config

CFG = load_config()


class ContextBuffer:
    def __init__(self):
        self.committed_words: list[str] = []
        self.current_prefix: str = ""

    def on_char(self, char: str) -> None:
        raise NotImplementedError("Phase 7")

    def on_word_boundary(self) -> None:
        raise NotImplementedError("Phase 7")

    def reset(self) -> None:
        raise NotImplementedError("Phase 7")

    def snapshot(self) -> tuple[tuple[str, ...], str]:
        """Returns (last-two-words-as-context, current_prefix) for predictor.predict()."""
        raise NotImplementedError("Phase 7")


def start_listener(on_update) -> None:
    """Start the pynput global listener, calling on_update(buffer) per debounced keystroke."""
    raise NotImplementedError("Phase 7")
