# samuel-python-cosine-sim

A naive pure-Python cosine similarity search. The deliberately slow
baseline for an upcoming CodSpeed demo.

```bash
uv sync
uv run pytest
```

## Layout

- `src/samuel_python_cosine_sim/search.py` — `dot`, `norm`, `cosine_similarity`, `top_k_similar`
- `tests/test_search.py` — correctness tests, with numpy as an oracle
