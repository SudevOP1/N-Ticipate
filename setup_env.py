"""One-time environment setup for N-Ticipate.

Creates the data directories, downloads the NLTK corpora the later phases
need, and reports what is present. Safe to re-run — everything is idempotent.

    python setup_env.py            # create dirs + download corpora
    python setup_env.py --check    # report status only, download nothing
"""

from __future__ import annotations

import argparse
import sys

from nticipate.config import data_dir, load_config, resolve_path

# (nltk id, where it lands, what needs it)
NLTK_PACKAGES: list[tuple[str, str, str]] = [
    ("punkt", "tokenizers/punkt", "Phase 1 sentence splitting"),
    ("punkt_tab", "tokenizers/punkt_tab", "Phase 1 sentence splitting (NLTK >= 3.8.2)"),
    ("brown", "corpora/brown", "Phase 2 English training corpus"),
    ("treebank", "corpora/treebank", "Phase 4 tagged English corpus"),
    ("universal_tagset", "taggers/universal_tagset", "Phase 4 universal tagset"),
    ("indian", "corpora/indian", "Phase 5 tagged Hindi corpus"),
    ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger",
     "Phase 4 baseline comparison only"),
]

DATA_DIRS = ["raw", "processed", "models"]


def ensure_dirs() -> None:
    for kind in DATA_DIRS:
        path = data_dir(kind, create=True)
        print(f"  [dir ] {path}")
    print(f"  [dir ] {resolve_path(load_config()['paths']['report'], create=True)}")


def _is_present(nltk_module, resource: str) -> bool:
    try:
        nltk_module.data.find(resource)
        return True
    except LookupError:
        return False


def check_nltk(download: bool) -> int:
    try:
        import nltk
    except ImportError:
        print("  [FAIL] nltk not installed -- run: pip install -r requirements.txt")
        return 1

    missing = 0
    for pkg_id, resource, why in NLTK_PACKAGES:
        if _is_present(nltk, resource):
            print(f"  [ ok ] {pkg_id:<28} ({why})")
            continue
        if not download:
            print(f"  [MISS] {pkg_id:<28} ({why})")
            missing += 1
            continue
        print(f"  [get ] {pkg_id} ...")
        nltk.download(pkg_id, quiet=True)
        if _is_present(nltk, resource):
            print(f"  [ ok ] {pkg_id}")
        else:
            # Not fatal: Phase 1 has a regex fallback, and the Hindi corpus
            # can be supplied by hand as a .conll file under data/raw.
            print(f"  [WARN] {pkg_id} still unavailable -- a fallback will be used")
            missing += 1
    return missing


def check_imports() -> int:
    required = ["yaml", "nltk", "numpy"]
    optional = {
        "pynput": "Phase 7 global keystroke hook",
        "pystray": "Phase 7 tray icon",
        "PIL": "Phase 7 tray icon rendering",
        "matplotlib": "notebook plots",
    }
    failed = 0
    for name in required:
        try:
            __import__(name)
            print(f"  [ ok ] {name}")
        except ImportError:
            print(f"  [FAIL] {name} missing -- pip install -r requirements.txt")
            failed += 1
    for name, why in optional.items():
        try:
            __import__(name)
            print(f"  [ ok ] {name:<12} ({why})")
        except ImportError:
            print(f"  [skip] {name:<12} ({why}) -- not installed")
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up the N-Ticipate environment.")
    parser.add_argument("--check", action="store_true",
                        help="report status only, download nothing")
    args = parser.parse_args()

    print(f"Python {sys.version.split()[0]}")

    print("\nConfig")
    cfg = load_config()
    print(f"  [ ok ] {cfg['project']['name']} v{cfg['project']['version']}")

    print("\nDirectories")
    ensure_dirs()

    print("\nPackages")
    failed = check_imports()

    print("\nNLTK corpora")
    missing = check_nltk(download=not args.check)

    print("\n" + "-" * 60)
    if failed:
        print(f"{failed} required package(s) missing. "
              f"Run: pip install -r requirements.txt")
        return 1
    if missing:
        print(f"Setup usable, {missing} NLTK resource(s) unavailable "
              f"(fallbacks will be used).")
    else:
        print("Setup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
