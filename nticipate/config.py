"""Cached YAML configuration loader.

Every phase reads its tunables from ``config.yaml`` through this module, so no
module hardcodes a magic number and the whole system can be re-tuned from one
file.

    from nticipate.config import load_config, get

    cfg = load_config()
    order = get("ngram.max_order", 3)
    models_dir = resolve_path(get("paths.models"))
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

#: Project root — this file lives at <root>/nticipate/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Default config location, overridable with the NTICIPATE_CONFIG env var.
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def config_path() -> Path:
    """Return the config file path, honouring ``NTICIPATE_CONFIG``."""
    override = os.environ.get("NTICIPATE_CONFIG")
    return Path(override).expanduser().resolve() if override else DEFAULT_CONFIG_PATH


@lru_cache(maxsize=8)
def _load_cached(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {path}. Expected it at the project root, "
            f"or point NTICIPATE_CONFIG at it."
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")
    return data


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and cache the config.

    Repeated calls are free — the parsed dict is memoised per path. Call
    :func:`reload_config` after editing the file in a long-running session
    (the tray app does this when settings change).
    """
    target = Path(path).expanduser().resolve() if path else config_path()
    return _load_cached(str(target))


def reload_config() -> dict[str, Any]:
    """Drop the cache and re-read from disk."""
    _load_cached.cache_clear()
    return load_config()


def get(key: str, default: Any = None, cfg: dict[str, Any] | None = None) -> Any:
    """Look up a dotted key, e.g. ``get("prediction.personalization.enabled")``.

    Returns ``default`` if any segment of the path is missing.
    """
    node: Any = cfg if cfg is not None else load_config()
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def require(key: str, cfg: dict[str, Any] | None = None) -> Any:
    """Like :func:`get`, but raise if the key is absent.

    Used for values with no sensible default, where silently falling back
    would hide a config mistake behind plausible-looking output.
    """
    sentinel = object()
    value = get(key, sentinel, cfg)
    if value is sentinel:
        raise KeyError(f"Required config key missing: {key!r}")
    return value


def resolve_path(value: str | Path, create: bool = False) -> Path:
    """Resolve a config path against the project root.

    Absolute paths are returned as-is. With ``create=True`` the directory is
    created (for a file path, its parent).
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if create:
        directory = path if not path.suffix else path.parent
        directory.mkdir(parents=True, exist_ok=True)
    return path


def data_dir(kind: str = "raw", create: bool = False) -> Path:
    """Return one of the data directories: ``raw``, ``processed`` or ``models``."""
    keys = {
        "raw": "paths.data_raw",
        "processed": "paths.data_processed",
        "models": "paths.models",
    }
    if kind not in keys:
        raise ValueError(f"Unknown data dir {kind!r}; expected one of {sorted(keys)}")
    return resolve_path(require(keys[kind]), create=create)
