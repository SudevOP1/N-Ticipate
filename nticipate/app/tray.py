"""
Phase 7: system tray icon (pystray).

Menu: enable/disable, language toggle (English / regional), settings,
"retrain on my typing" (flushes UserProfile), quit.
"""


def build_tray_icon():
    """Return a pystray.Icon wired to the app's enable/disable/quit state."""
    raise NotImplementedError("Phase 7")


def run() -> None:
    """Entry point: python -m nticipate.app.tray"""
    raise NotImplementedError("Phase 7")


if __name__ == "__main__":
    run()
