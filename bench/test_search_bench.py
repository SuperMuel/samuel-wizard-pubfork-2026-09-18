"""Benchmarks for the cosine similarity search."""

from __future__ import annotations

import random

import pytest

from samuel_python_cosine_sim.search import cosine_similarity, dot, norm, top_k_similar


def make_vector(dim: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(dim)]


def make_corpus(n_docs: int, dim: int, seed: int = 0) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(n_docs)]


def test_dot(benchmark):
    a, b = make_vector(1024, seed=1), make_vector(1024, seed=2)
    assert isinstance(benchmark(dot, a, b), float)


def test_norm(benchmark):
    a = make_vector(1024, seed=3)
    assert benchmark(norm, a) > 0


def test_cosine_similarity(benchmark):
    a, b = make_vector(384, seed=4), make_vector(384, seed=5)
    assert -1.0 <= benchmark(cosine_similarity, a, b) <= 1.0


def test_top_k_similar(benchmark):
    docs = make_corpus(500, 384, seed=6)
    query = make_vector(384, seed=7)
    assert len(benchmark(top_k_similar, query, docs, 10)) == 10
