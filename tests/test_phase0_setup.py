"""
Phase 0 smoke tests. These should pass right now, before any other
phase is implemented -- they only test the scaffolding itself.

Run with: pytest tests/test_phase0_setup.py -v
"""

import os
from nticipate.config import load_config, resolve_path


def test_config_loads():
    cfg = load_config()
    assert "ngram" in cfg
    assert "hmm" in cfg
    assert "app" in cfg


def test_config_has_expected_ngram_settings():
    cfg = load_config()
    assert cfg["ngram"]["orders"] == [1, 2, 3]
    assert cfg["ngram"]["smoothing"] in {"mle", "laplace", "stupid_backoff", "kneser_ney"}


def test_data_folders_exist():
    cfg = load_config()
    for rel_path in cfg["paths"].values():
        assert os.path.isdir(resolve_path(rel_path)), f"missing folder: {rel_path}"


def test_package_imports():
    # These should import without error even though functions inside
    # raise NotImplementedError -- that's expected until later phases.
    import nticipate.preprocess       # noqa: F401
    import nticipate.ngram            # noqa: F401
    import nticipate.trie             # noqa: F401
    import nticipate.hmm              # noqa: F401
    import nticipate.predictor        # noqa: F401
    import nticipate.userprofile      # noqa: F401
