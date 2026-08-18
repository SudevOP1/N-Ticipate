"""
Phase 7: injects the accepted suggestion into the focused application.

Two strategies:
    - pynput keyboard controller: types remaining characters directly
    - clipboard + paste: faster for long completions, but must save and
      restore the user's existing clipboard contents afterward
"""


def inject_via_typing(remaining_text: str) -> None:
    raise NotImplementedError("Phase 7")


def inject_via_clipboard(remaining_text: str) -> None:
    """Must snapshot and restore the existing clipboard -- never clobber it silently."""
    raise NotImplementedError("Phase 7")
