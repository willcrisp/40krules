"""Corpus-driven spell correction for query preprocessing (SymSpell-style).

Deliberately NOT a dictionary spell checker: the vocabulary is built from the
ingested rules text, so domain terms are never "corrected" into plain
English. Only out-of-vocabulary tokens are touched, which means real rule
words ("embark") can never be rewritten into near neighbours ("disembark").
Lookup runs against a precomputed deletion index — well under a millisecond,
no ML in the hot path.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9]+")

MIN_TOKEN_LEN = 4  # don't second-guess very short tokens ("d6", "ap", "los")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _deletes(word: str, max_edits: int) -> set[str]:
    """All strings reachable from `word` by deleting up to max_edits chars."""
    results = {word}
    frontier = {word}
    for _ in range(max_edits):
        nxt: set[str] = set()
        for w in frontier:
            for i in range(len(w)):
                nxt.add(w[:i] + w[i + 1 :])
        results |= nxt
        frontier = nxt
    return results


def _osa_distance(a: str, b: str, cap: int) -> int:
    """Optimal string alignment distance (Levenshtein + transpositions), capped."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev2: list[int] | None = None
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                cur[j] = min(cur[j], prev2[j - 2] + 1)  # type: ignore[index]
        if min(cur) > cap:
            return cap + 1
        prev2, prev = prev, cur
    return prev[-1]


@dataclass
class Correction:
    original: str
    corrected: str  # normalized token string, e.g. "disembark rules"
    changed: bool
    replacements: dict[str, str]  # original token -> corrected token


class SpellCorrector:
    def __init__(self, vocab: dict[str, int], max_edit_distance: int = 2) -> None:
        self.vocab = vocab
        self.max_edit_distance = max_edit_distance
        self._sorted_terms = sorted(vocab)
        self._delete_index: dict[str, list[str]] = {}
        for term in vocab:
            if len(term) < MIN_TOKEN_LEN:
                continue
            for d in _deletes(term, max_edit_distance):
                self._delete_index.setdefault(d, []).append(term)

    def _is_vocab_prefix(self, token: str) -> bool:
        idx = bisect_left(self._sorted_terms, token)
        return idx < len(self._sorted_terms) and self._sorted_terms[idx].startswith(token)

    def _known(self, token: str) -> bool:
        """In vocabulary, or a simple inflection of a vocabulary word.

        FTS porter-stems inflections to the same token anyway, so "rules" must
        not be "corrected" just because the corpus only says "rule".
        """
        if token in self.vocab:
            return True
        for suffix in ("s", "es", "d", "ed", "ing"):
            if token.endswith(suffix) and token[: -len(suffix)] in self.vocab:
                return True
        return f"{token}s" in self.vocab

    def _lookup(self, token: str, max_ed: int) -> str | None:
        candidates: set[str] = set()
        for d in _deletes(token, max_ed):
            candidates.update(self._delete_index.get(d, ()))
        best: tuple[tuple[int, int, str], str] | None = None
        for term in candidates:
            dist = _osa_distance(token, term, max_ed)
            if dist <= max_ed:
                key = (dist, -self.vocab[term], term)
                if best is None or key < best[0]:
                    best = (key, term)
        return best[1] if best else None

    def correct_token(self, token: str, is_last: bool = False) -> str:
        """Return the token, corrected only when it is out-of-vocabulary."""
        if len(token) < MIN_TOKEN_LEN or token.isdigit() or self._known(token):
            return token
        # The last token is usually still being typed: if it's a prefix of a
        # known term, leave it alone — FTS5 prefix matching handles it.
        if is_last and self._is_vocab_prefix(token):
            return token
        max_ed = min(self.max_edit_distance, 1 if len(token) <= 5 else 2)
        return self._lookup(token, max_ed) or token

    def correct_query(self, query: str) -> Correction:
        tokens = tokenize(query)
        replacements: dict[str, str] = {}
        out: list[str] = []
        for i, tok in enumerate(tokens):
            fixed = self.correct_token(tok, is_last=(i == len(tokens) - 1))
            out.append(fixed)
            if fixed != tok:
                replacements[tok] = fixed
        corrected = " ".join(out)
        return Correction(
            original=query,
            corrected=corrected,
            changed=bool(replacements),
            replacements=replacements,
        )
