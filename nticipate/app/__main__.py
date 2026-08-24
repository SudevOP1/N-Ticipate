"""Entry point for the Phase 7 desktop app.

    python -m nticipate.app             # tray icon + global hook
    python -m nticipate.app --editor    # the fallback editor window
    python -m nticipate.app --check     # report what is available, run nothing

``--check`` exists because the two hard parts of this phase fail in ways that
look like the app being broken: ``pynput`` missing, or no packaged model built
yet. It answers both without starting a keyboard hook.
"""

from __future__ import annotations

import argparse
import sys


def check() -> int:
    """Print what the desktop layer can and cannot do on this machine."""
    from nticipate.app import win32
    from nticipate.app.hooks import KeystrokeHook
    from nticipate.app.overlay import SuggestionOverlay
    from nticipate.app.tray import TrayIcon, model_paths

    print(f"Platform            {sys.platform} "
          f"({'supported' if win32.IS_WINDOWS else 'hook/overlay unavailable'})")
    print(f"pynput (hook)       {'ok' if KeystrokeHook.available() else 'MISSING'}")
    print(f"pystray + Pillow    {'ok' if TrayIcon.available() else 'MISSING'}")
    print(f"tkinter (overlay)   {'ok' if SuggestionOverlay.available() else 'MISSING'}")

    print("\nModels")
    paths = model_paths()
    missing = 0
    for name in ("ngram", "corpus", "tagger", "tagger_hindi"):
        path = paths[name]
        if path is None:
            print(f"  [MISS] {name:<12} not built")
            missing += name == "ngram"
        else:
            print(f"  [ ok ] {name:<12} {path.name} "
                  f"({path.stat().st_size / 1e6:.1f} MB)")
    profile = paths["profile"]
    print(f"  [{' ok ' if profile.exists() else 'none'}] {'profile':<12} {profile}")

    if missing:
        print("\nNo n-gram model: run notebooks/02_ngram_models.ipynb first.")
        return 1
    if not KeystrokeHook.available():
        print("\npynput missing: the tray app cannot run, but --editor can.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m nticipate.app",
        description="N-Ticipate — n-gram + HMM autocomplete in the system tray.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--editor", action="store_true",
                      help="run the fallback editor window instead of the tray app")
    mode.add_argument("--check", action="store_true",
                      help="report availability of the desktop dependencies and models")
    parser.add_argument("--learn", action="store_true",
                        help="learn from typing this session (off by default)")
    args = parser.parse_args(argv)

    if args.check:
        return check()
    if args.editor:
        from nticipate.app.editor import run

        return run(learning=True if args.learn else None)

    from nticipate.app.tray import run

    return run(learning=True if args.learn else None)


if __name__ == "__main__":
    raise SystemExit(main())
