"""
Loads config.yaml once and hands back a plain dict.

Usage:
    from nticipate.config import load_config
    cfg = load_config()
    cfg["ngram"]["smoothing"]      # "stupid_backoff"
    cfg["paths"]["processed_dir"]  # "data/processed"

Every module in every phase should read parameters from here rather than
hardcoding them -- that's the whole point of Phase 0.
"""

import os
import functools
import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")


@functools.lru_cache(maxsize=1)
def load_config(path: str = _DEFAULT_CONFIG_PATH) -> dict:
    """Load and cache config.yaml. Cached so repeated calls are free."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> str:
    """Turn a config-relative path (e.g. cfg['paths']['processed_dir'])
    into an absolute path rooted at the project directory."""
    return os.path.join(_PROJECT_ROOT, relative_path)


if __name__ == "__main__":
    # Quick sanity check: `python -m nticipate.config`
    cfg = load_config()
    print("Loaded config keys:", list(cfg.keys()))
    print("Processed dir resolves to:", resolve_path(cfg["paths"]["processed_dir"]))
