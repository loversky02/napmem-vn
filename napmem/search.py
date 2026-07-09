from __future__ import annotations

import math
import re
from collections.abc import Iterable

_WORD = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def keyword_score(query: str, text: str) -> float:
    q = set(tokenize(query))
    toks = tokenize(text)
    if not q or not toks:
        return 0.0
    overlap = sum(1 for tok in toks if tok in q)
    return overlap / math.sqrt(len(toks))


def rank_keyword(query: str, items: Iterable[tuple[str, str]], limit: int) -> list[tuple[str, float]]:
    scored = [(item_id, keyword_score(query, text)) for item_id, text in items]
    scored = [(item_id, score) for item_id, score in scored if score > 0]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:limit]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores
