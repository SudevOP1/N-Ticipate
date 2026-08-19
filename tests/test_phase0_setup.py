"""Phase 0 smoke tests.

These pass before any real NLP code exists. Their job is to catch the boring
failures early: a missing directory, a typo'd config key that a later phase
would only discover halfway through a training run.
"""

from __future__ import annotations

import textwrap

import pytest

from nticipate import __version__
from nticipate.config import (
    PROJECT_ROOT,
    data_dir,
    get,
    load_config,
    reload_config,
    require,
    resolve_path,
)


# --------------------------------------------------------------- structure

def test_project_root_is_repo_root():
    assert (PROJECT_ROOT / "config.yaml").is_file()
    assert (PROJECT_ROOT / "PLAN.md").is_file()


@pytest.mark.parametrize(
    "relative",
    ["nticipate", "nticipate/app", "tests", "notebooks", "report",
     "data/raw", "data/processed", "data/models"],
)
def test_expected_directories_exist(relative):
    assert (PROJECT_ROOT / relative).is_dir(), f"missing directory: {relative}"


def test_package_imports():
    assert __version__


# ------------------------------------------------------------ config load

def test_config_loads():
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert cfg["project"]["name"] == "N-Ticipate"


def test_config_is_cached():
    assert load_config() is load_config()


def test_reload_returns_equal_config():
    first = load_config()
    assert reload_config() == first


# ------------------------------------------------- keys later phases read

# Every key a later phase depends on. Adding a phase means adding its keys
# here first — that is what keeps "done when: returns every key later phases
# read" honest rather than aspirational.
REQUIRED_KEYS = [
    # paths
    "paths.data_raw", "paths.data_processed", "paths.models",
    "paths.user_profile", "paths.report",
    # Phase 1
    "preprocessing.remove_stopwords", "preprocessing.remove_punctuation",
    "preprocessing.lowercase_for_counts", "preprocessing.truecase_output",
    "preprocessing.tokenizer", "preprocessing.sentence_splitter",
    "preprocessing.min_token_freq", "preprocessing.max_vocab_size",
    "preprocessing.unk_token", "preprocessing.bos_token",
    "preprocessing.eos_token", "preprocessing.normalize_unicode",
    "preprocessing.split.train", "preprocessing.split.dev",
    "preprocessing.split.test", "preprocessing.split.seed",
    # Phase 2
    "ngram.max_order", "ngram.smoothing", "ngram.laplace_k",
    "ngram.backoff_alpha", "ngram.kneser_ney_discount",
    "ngram.pruning.enabled", "ngram.pruning.min_count",
    "ngram.pruning.max_continuations", "ngram.corpus.english",
    # Phase 3
    "prediction.max_suggestions", "prediction.min_prefix_len",
    "prediction.candidate_pool", "prediction.personalization.enabled",
    "prediction.personalization.lambda_max",
    "prediction.personalization.lambda_growth_tokens",
    "prediction.latency.debounce_ms", "prediction.latency.p95_budget_ms",
    # Phases 4-5
    "hmm.language", "hmm.tagset", "hmm.smoothing_k", "hmm.use_log_space",
    "hmm.unknown_word_strategy", "hmm.corpora.english", "hmm.corpora.hindi",
    "hmm.split.train", "hmm.split.test", "hmm.split.seed",
    # Phase 6
    "reranking.enabled", "reranking.alpha", "reranking.tag_context_size",
    # Phase 7
    "app.hotkeys.accept", "app.hotkeys.dismiss", "app.overlay.max_items",
    "app.capture.max_context_words", "app.capture.injection_method",
    "app.privacy.disable_in_password_fields",
    "app.privacy.log_keystrokes_to_disk", "app.privacy.telemetry",
    # Phase 8
    "evaluation.hit_at_k", "evaluation.latency_percentiles",
]


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_required_key_present(key):
    require(key)  # raises KeyError if missing


def test_get_returns_default_for_missing_key():
    assert get("nope.not.here", "fallback") == "fallback"


def test_require_raises_for_missing_key():
    with pytest.raises(KeyError):
        require("nope.not.here")


# ---------------------------------------------------------- value sanity

def test_ngram_order_supports_trigrams():
    # The coursework requires bigrams and trigrams explicitly.
    assert require("ngram.max_order") >= 3


def test_smoothing_method_is_implemented():
    assert require("ngram.smoothing") in {
        "mle", "laplace", "stupid_backoff", "kneser_ney"
    }


def test_preprocessing_keeps_stopwords_and_punctuation():
    # Removing them is right for classification, wrong for language
    # modelling — "of the" is exactly what a bigram model must predict.
    assert require("preprocessing.remove_stopwords") is False
    assert require("preprocessing.remove_punctuation") is False


def test_corpus_split_sums_to_one():
    split = require("preprocessing.split")
    assert abs(split["train"] + split["dev"] + split["test"] - 1.0) < 1e-9


def test_hmm_split_sums_to_one():
    split = require("hmm.split")
    assert abs(split["train"] + split["test"] - 1.0) < 1e-9


def test_hmm_uses_log_space_viterbi():
    # Linear-space Viterbi underflows on real sentence lengths.
    assert require("hmm.use_log_space") is True


def test_reranking_alpha_in_range():
    assert 0.0 <= require("reranking.alpha") <= 1.0


def test_personalization_lambda_in_range():
    assert 0.0 <= require("prediction.personalization.lambda_max") <= 1.0


def test_latency_budget_matches_debounce():
    # A suggestion that arrives after the next keystroke is useless.
    assert require("prediction.latency.p95_budget_ms") <= \
        require("prediction.latency.debounce_ms") * 2


def test_privacy_defaults_are_safe():
    # Phase 7 hard requirements: nothing leaves the machine, nothing is
    # written to disk, password fields are never captured.
    assert require("app.privacy.telemetry") is False
    assert require("app.privacy.log_keystrokes_to_disk") is False
    assert require("app.privacy.disable_in_password_fields") is True


# ------------------------------------------------------------ path helpers

def test_resolve_path_makes_relative_absolute():
    assert resolve_path("data/raw").is_absolute()


def test_resolve_path_leaves_absolute_alone(tmp_path):
    assert resolve_path(tmp_path) == tmp_path.resolve()


@pytest.mark.parametrize("kind", ["raw", "processed", "models"])
def test_data_dirs_exist(kind):
    assert data_dir(kind).is_dir()


def test_data_dir_rejects_unknown_kind():
    with pytest.raises(ValueError):
        data_dir("nonsense")


# ------------------------------------------------------- override support

def test_config_path_override(tmp_path):
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        textwrap.dedent(
            """
            project:
              name: Custom
            ngram:
              max_order: 5
            """
        ),
        encoding="utf-8",
    )
    cfg = load_config(custom)
    assert get("ngram.max_order", cfg=cfg) == 5
    # The default config is untouched by the override.
    assert require("project.name") == "N-Ticipate"


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")
