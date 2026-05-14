from __future__ import annotations

import math
from collections.abc import Sequence


Vector = Sequence[float]


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def norm(a: Vector) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: Vector, b: Vector) -> float:
    return dot(a, b) / (norm(a) * norm(b))


def top_k_similar(
    query: Vector,
    docs: Sequence[Vector],
    k: int,
) -> list[tuple[int, float]]:
    """Return the ``k`` most similar docs to ``query`` as ``(index, score)`` pairs.

    Sorted in descending order of cosine similarity.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    scores = [(i, cosine_similarity(query, doc)) for i, doc in enumerate(docs)]
    scores.sort(key=lambda pair: pair[1], reverse=True)
    return scores[:k]
