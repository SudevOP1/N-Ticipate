"""
Phase 0 setup script.

Run once after cloning:
    python setup_env.py

What it does:
    1. Confirms it's running inside a virtualenv (won't touch system Python).
    2. Downloads the NLTK corpora every later phase depends on.
    3. Verifies config.yaml parses and required folders exist.

This does NOT install pip packages -- do that first with:
    python -m venv .venv
    .venv\\Scripts\\activate        (Windows)
    source .venv/bin/activate      (macOS/Linux)
    pip install -r requirements.txt
    python setup_env.py
"""

import sys
import os
import yaml

REQUIRED_NLTK_CORPORA = [
    "brown",          # Phase 1/2: general English corpus
    "gutenberg",       # Phase 1/2: general English corpus
    "reuters",         # Phase 1/2: general English corpus
    "punkt",           # Phase 1: sentence tokenizer
    "punkt_tab",        # Phase 1: newer NLTK tokenizer data
    "treebank",         # Phase 4: POS-tagged English
    "universal_tagset", # Phase 4: coarse 12-tag set
    "indian",           # Phase 5: Hindi/Marathi tagged corpus
]


def check_virtualenv() -> None:
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        print(
            "[warn] Doesn't look like a virtualenv is active.\n"
            "       Recommended: python -m venv .venv && activate it, "
            "then re-run this script."
        )
    else:
        print(f"[ok] Running inside virtualenv: {sys.prefix}")


def download_nltk_data() -> None:
    import nltk

    for corpus in REQUIRED_NLTK_CORPORA:
        try:
            print(f"[nltk] downloading '{corpus}'...")
            nltk.download(corpus, quiet=True)
        except Exception as exc:  # noqa: BLE001 -- report and continue
            print(f"[warn] could not download '{corpus}': {exc}")


def check_config() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(here, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key in ("project", "paths", "preprocessing", "ngram", "predictor", "hmm", "reranker", "app"):
        assert key in cfg, f"config.yaml is missing top-level key: {key}"

    for rel_path in cfg["paths"].values():
        full = os.path.join(here, rel_path)
        os.makedirs(full, exist_ok=True)

    print("[ok] config.yaml parsed and data folders present")


def main() -> None:
    check_virtualenv()
    check_config()
    download_nltk_data()
    print("\nPhase 0 setup complete. Next: implement nticipate/preprocess.py (Phase 1).")


if __name__ == "__main__":
    main()
