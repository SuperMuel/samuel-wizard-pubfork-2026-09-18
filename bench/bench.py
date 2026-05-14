import random
import time

from samuel_python_cosine_sim import top_k_similar


def make_corpus(n_docs: int, dim: int, seed: int = 0) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(n_docs)]


def main() -> None:
    n_docs, dim, k = 5_000, 384, 10
    rounds = 5
    docs = make_corpus(n_docs, dim)
    query = docs[0]

    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        top_k_similar(query, docs, k=k)
        times.append(time.perf_counter() - t0)

    times.sort()
    print(
        f"top_k_similar(n_docs={n_docs}, dim={dim}, k={k}) "
        f"over {rounds} rounds: "
        f"min={times[0] * 1000:.1f} ms, "
        f"median={times[len(times) // 2] * 1000:.1f} ms, "
        f"max={times[-1] * 1000:.1f} ms"
    )


if __name__ == "__main__":
    main()
