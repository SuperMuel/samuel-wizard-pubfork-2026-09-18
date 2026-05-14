import math

import numpy as np
import pytest

from samuel_python_cosine_sim.search import (
    cosine_similarity,
    dot,
    norm,
    top_k_similar,
)


def test_dot_basic():
    assert dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(32.0)


def test_norm_basic():
    assert norm([3.0, 4.0]) == pytest.approx(5.0)


def test_cosine_identical_vectors_is_one():
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors_is_minus_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]) == pytest.approx(-1.0)


def test_cosine_is_symmetric():
    a, b = [0.1, 0.7, -0.3, 0.4], [-0.2, 0.5, 0.9, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a))


def test_cosine_matches_numpy():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(128), rng.standard_normal(128)
    expected = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cosine_similarity(a.tolist(), b.tolist()) == pytest.approx(expected)


def test_dot_zip_strict_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        dot([1.0, 2.0], [1.0, 2.0, 3.0])


def test_top_k_returns_k_results():
    rng = np.random.default_rng(0)
    docs = rng.standard_normal((50, 32)).tolist()
    query = rng.standard_normal(32).tolist()

    results = top_k_similar(query, docs, k=5)

    assert len(results) == 5


def test_top_k_results_are_sorted_descending():
    rng = np.random.default_rng(1)
    docs = rng.standard_normal((30, 16)).tolist()
    query = rng.standard_normal(16).tolist()

    results = top_k_similar(query, docs, k=10)
    scores = [score for _, score in results]

    assert scores == sorted(scores, reverse=True)


def test_top_k_indices_are_unique_and_in_range():
    rng = np.random.default_rng(2)
    n_docs = 20
    docs = rng.standard_normal((n_docs, 8)).tolist()
    query = rng.standard_normal(8).tolist()

    results = top_k_similar(query, docs, k=7)
    indices = [i for i, _ in results]

    assert len(set(indices)) == len(indices)
    assert all(0 <= i < n_docs for i in indices)


def test_top_k_matches_numpy_reference():
    rng = np.random.default_rng(42)
    docs_np = rng.standard_normal((40, 24))
    query_np = rng.standard_normal(24)

    docs_norm = docs_np / np.linalg.norm(docs_np, axis=1, keepdims=True)
    query_norm = query_np / np.linalg.norm(query_np)
    expected_scores = docs_norm @ query_norm
    expected_top = sorted(
        enumerate(expected_scores.tolist()), key=lambda p: p[1], reverse=True
    )[:5]

    results = top_k_similar(query_np.tolist(), docs_np.tolist(), k=5)

    assert [i for i, _ in results] == [i for i, _ in expected_top]
    for (_, got), (_, exp) in zip(results, expected_top, strict=True):
        assert got == pytest.approx(exp)


def test_top_k_query_finds_itself():
    rng = np.random.default_rng(3)
    docs = rng.standard_normal((25, 12)).tolist()
    query = docs[7]

    results = top_k_similar(query, docs, k=1)

    assert results[0][0] == 7
    assert results[0][1] == pytest.approx(1.0)


def test_top_k_zero_returns_empty():
    docs = [[1.0, 0.0], [0.0, 1.0]]
    assert top_k_similar([1.0, 1.0], docs, k=0) == []


def test_top_k_larger_than_corpus_returns_all():
    docs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    results = top_k_similar([1.0, 0.0], docs, k=10)
    assert len(results) == 3


def test_top_k_negative_k_raises():
    with pytest.raises(ValueError):
        top_k_similar([1.0, 0.0], [[1.0, 0.0]], k=-1)


def test_cosine_normalization_invariance():
    a = [1.0, 2.0, 3.0]
    scaled = [10.0, 20.0, 30.0]
    other = [0.3, -0.1, 0.9]
    assert cosine_similarity(a, other) == pytest.approx(cosine_similarity(scaled, other))


def test_cosine_handles_negative_components():
    a, b = [-1.0, 2.0, -3.0], [1.0, -2.0, 3.0]
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_norm_of_unit_vector():
    assert norm([1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_cosine_with_known_angle():
    # angle of 60 degrees → cos = 0.5
    a = [1.0, 0.0]
    b = [math.cos(math.pi / 3), math.sin(math.pi / 3)]
    assert cosine_similarity(a, b) == pytest.approx(0.5)
