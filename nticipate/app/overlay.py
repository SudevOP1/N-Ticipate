"""
Phase 7: frameless, always-on-top, non-activating suggestion overlay.

Anchoring order of preference (CFG['app']['overlay']):
    1. caret position via GetGUIThreadInfo (Windows, ctypes) when available
    2. fallback: near mouse cursor

Must never steal focus (WS_EX_NOACTIVATE) -- that's the bug that makes
these apps unusable.
"""


class SuggestionOverlay:
    def __init__(self):
        self.visible = False

    def show(self, suggestions: list[str], anchor_pos: tuple[int, int]) -> None:
        raise NotImplementedError("Phase 7")

    def hide(self) -> None:
        raise NotImplementedError("Phase 7")

    def get_caret_position(self) -> tuple[int, int] | None:
        """Windows: ctypes call to GetGUIThreadInfo. Returns None if unsupported
        by the focused app, in which case caller should fall back to cursor pos.
        """
        raise NotImplementedError("Phase 7")
