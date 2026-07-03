"""Query spell correction: corpus-driven, additive, and never rewrites
in-vocabulary terms (so embark/disembark cannot swap via correction)."""

from __future__ import annotations

import pytest

from rulehound.config import SearchConfig
from rulehound.search.hybrid import hybrid_search
from rulehound.search.spell import SpellCorrector

CFG = SearchConfig()


@pytest.fixture(scope="module")
def corrector(ingested) -> SpellCorrector:
    from rulehound.store.sqlite_store import SqliteStore

    cfg, _ = ingested
    s = SqliteStore(cfg.paths.db_path, vector_dim=cfg.embedding.dimension)
    vocab = s.load_vocab()
    s.close()
    assert vocab, "ingest should have built a vocabulary"
    return SpellCorrector(vocab)


def test_vocab_built_from_corpus(corrector: SpellCorrector) -> None:
    assert "disembark" in corrector.vocab
    assert "transport" in corrector.vocab
    # commentary-only words are not in FTS and not in vocab either
    assert corrector.vocab["the"] > corrector.vocab["disembark"]


def test_corrects_oov_typos(corrector: SpellCorrector) -> None:
    assert corrector.correct_token("dismbark") == "disembark"
    assert corrector.correct_token("transprot") == "transport"  # transposition
    assert corrector.correct_token("engagment") == "engagement"


def test_never_rewrites_vocab_terms(corrector: SpellCorrector) -> None:
    # "embark" is in the corpus: it must never be corrected toward "disembark"
    assert corrector.correct_token("embark") == "embark"
    assert corrector.correct_token("disembark") == "disembark"


def test_short_and_numeric_tokens_untouched(corrector: SpellCorrector) -> None:
    assert corrector.correct_token("d6") == "d6"
    assert corrector.correct_token("teh") == "teh"  # len < 4: leave it alone
    assert corrector.correct_token("2026") == "2026"


def test_last_token_prefix_not_corrected(corrector: SpellCorrector) -> None:
    # mid-typing: "disemb" is a prefix of "disembark(s)" — leave for FTS prefix
    assert corrector.correct_token("disemb", is_last=True) == "disemb"
    c = corrector.correct_query("rules for disemb")
    assert not c.changed


def test_correct_query_reports_replacements(corrector: SpellCorrector) -> None:
    c = corrector.correct_query("dismbark from a transprot")
    assert c.changed
    assert c.replacements == {"dismbark": "disembark", "transprot": "transport"}
    assert "disembark" in c.corrected and "transport" in c.corrected

    clean = corrector.correct_query("disembark rules")
    assert not clean.changed
    assert clean.replacements == {}


def test_hybrid_search_with_typos(store, corrector: SpellCorrector) -> None:
    def top1(q: str) -> str:
        results, _, _ = hybrid_search(store, None, q, CFG, corrector=corrector)
        assert results, f"no results for {q!r}"
        return results[0].rule_id.split(".")[-1]

    assert top1("dismbark rules") == "disembark"
    assert top1("engagment range") == "engagement-range"
    assert top1("deep strke") == "deep-strike"
    # adversarial pair stays safe with the corrector active
    assert top1("embark") == "embark"


def test_correction_surfaced_to_caller(store, corrector: SpellCorrector) -> None:
    _, _, correction = hybrid_search(store, None, "dismbark rules", CFG, corrector=corrector)
    assert correction is not None and correction.changed
    assert correction.corrected == "disembark rules"

    _, _, correction = hybrid_search(store, None, "disembark rules", CFG, corrector=corrector)
    assert correction is not None and not correction.changed
