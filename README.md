# samuel-python-cosine-sim

A pure-Python cosine similarity search.

```bash
uv sync
uv run pytest
```

## Layout

- `src/samuel_python_cosine_sim/search.py` — `dot`, `norm`, `cosine_similarity`, `top_k_similar`
- `tests/test_search.py` — correctness tests, with numpy as an oracle
