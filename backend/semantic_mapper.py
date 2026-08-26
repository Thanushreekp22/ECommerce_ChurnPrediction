"""
backend/semantic_mapper.py

Lightweight, fully local "AI" column-name matcher used by feature_mapper.

Canonical feature mapping no longer depends exclusively on a static alias
lookup table. This module maps uploaded column names to canonical churn
features even when there is no exact alias, using an explainable composite
similarity:

    score = 0.50 * tfidf_cosine + 0.30 * token_semantic + 0.20 * edit_alias

- tfidf_cosine : word-level TF-IDF alignment with the concept vocabulary
- token_semantic: fuzzy per-token match (so "orders"~"order",
                  "registered"~"registration", "joined"~"join")
- edit_alias   : closest alias similarity, so typos still score well

A column is promoted only when the best candidate clears an absolute
threshold and beats the second-best by a clear margin.  Unrelated columns
(e.g. "product", "category", "color") stay unmapped because they share no
meaningful vocabulary with any concept.
"""

import re
from difflib import SequenceMatcher

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.canonical_features import CANONICAL_FEATURES


# Semantic descriptors for each canonical feature, going beyond the alias text.
CANONICAL_CONCEPTS: dict[str, list[str]] = {
    "customer_id": [
        "customer", "client", "user", "buyer", "member", "account",
        "identifier", "profile", "subscriber", "email", "handle", "owner",
    ],
    "total_orders": [
        "order", "orders", "purchase", "purchases", "transaction",
        "transactions", "count", "frequency", "items", "booking", "units",
    ],
    "total_spend": [
        "spend", "spent", "revenue", "money", "amount", "value", "sales",
        "lifetime", "ltv", "monetary", "cashback", "total", "sum", "gmv",
    ],
    "last_purchase_date": [
        "last", "latest", "recent", "purchase", "order", "transaction",
        "date", "activity", "active",
    ],
    "signup_date": [
        "signup", "sign-up", "registration", "register", "join", "joined", "since", "member-since", "first-joined",
        "created", "creation", "member", "member-since", "enroll", "start",
        "joined-date", "first-order", "registration-date", "signup-date",
    ],
    "age": [
        "age", "customer-age", "years-old", "years", "personal-age", "age-years", "client", "buyer-age",
    ],
}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+[0-9]*", str(text).strip().lower())


def _concept_text(name: str) -> str:
    parts = [name] + list(CANONICAL_FEATURES[name].get("aliases", [])) + CANONICAL_CONCEPTS.get(name, [])
    return " ".join(parts)


_CONCEPT_TEXTS = {name: _concept_text(name) for name in CANONICAL_FEATURES}
_ORDERED = sorted(_CONCEPT_TEXTS.keys())
_VECTORIZER = TfidfVectorizer(analyzer="word", token_pattern=r"[a-z0-9]+")
_MATRIX = _VECTORIZER.fit_transform([_CONCEPT_TEXTS[name] for name in _ORDERED])
_CONCEPT_WORDS = {name: set(_words(_CONCEPT_TEXTS[name])) for name in CANONICAL_FEATURES}
_CONCEPT_ALIASES = {name: list(CANONICAL_FEATURES[name].get("aliases", [])) for name in CANONICAL_FEATURES}


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _token_semantic(query_words: list[str], concept_words: set[str]) -> float:
    if not query_words:
        return 0.0
    scores, weights = [], []
    for word in query_words:
        best = max((_ratio(word, cw) for cw in concept_words), default=0.0)
        scores.append(best)
        weights.append(max(len(word), 1))
    total = sum(weights)
    return sum(score * weight for score, weight in zip(scores, weights)) / total if total else 0.0


def _best_alias_edit(column_text: str, aliases: list[str]) -> float:
    return max((_ratio(column_text, alias) for alias in aliases), default=0.0)


def semantic_column_map(columns) -> dict[str, dict]:
    if columns is None or len(list(columns)) == 0:
        return {}
    candidates: list[tuple[float, str, str]] = []
    query_vectors = _VECTORIZER.transform([str(c).strip().lower() for c in columns])

    for idx, column in enumerate(columns):
        column_text = str(column).strip().lower()
        query_words = _words(column_text)
        if not query_words:
            continue
        row = query_vectors[idx]
        for name in _ORDERED:
            cosine = float(cosine_similarity(row, _MATRIX[_ORDERED.index(name)])[0, 0])
            token_sem = _token_semantic(query_words, _CONCEPT_WORDS[name])
            edit_alias = _best_alias_edit(column_text, _CONCEPT_ALIASES[name])
            score = round(0.50 * cosine + 0.30 * token_sem + 0.20 * edit_alias, 4)
            candidates.append((score, column, name))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    accepted: dict[str, dict] = {}
    used: set[str] = set()
    for score, column, name in candidates:
        if column in accepted or name in used:
            continue
        other = max((cand[0] for cand in candidates if cand[1] == column and cand[2] != name), default=0.0)
        if score < 0.50:
            continue
        if other > 0 and score - other < (0.12 if score >= 0.55 else 0.05):
            continue
        accepted[column] = {
            "canonical": name, "confidence": round(score, 4), "reason": "semantic similarity",
        }
        used.add(name)
    return accepted
