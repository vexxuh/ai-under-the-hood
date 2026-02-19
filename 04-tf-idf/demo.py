"""Demo: chapter similarity and keyword extraction using TF-IDF."""

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_PATH = Path(__file__).parent / "tfidf_model.pkl"


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def main():
    print("Loading TF-IDF model...")
    model = load_model()
    tfidf = model["tfidf_matrix"]
    bow = model["bow_matrix"]
    idf = model["idf"]
    names = model["chapter_names"]
    idx_to_word = model["idx_to_word"]
    vocab = model["vocab"]

    print(f"Chapters: {len(names)}")
    print(f"Vocabulary: {len(vocab):,} words")

    # Keywords per chapter (what makes each chapter distinctive)
    print("\n" + "=" * 60)
    print("TOP 5 KEYWORDS PER CHAPTER (by TF-IDF)")
    print("=" * 60)
    for i, name in enumerate(names):
        top_idx = np.argsort(-tfidf[i])[:5]
        keywords = [idx_to_word[idx] for idx in top_idx]
        print(f"  {name}")
        print(f"    {', '.join(keywords)}")

    # Highest IDF words (rarest across chapters)
    print("\n" + "=" * 60)
    print("TOP 20 RAREST WORDS (highest IDF)")
    print("=" * 60)
    top_idf = np.argsort(-idf)[:20]
    for idx in top_idf:
        word = idx_to_word[idx]
        doc_freq = np.count_nonzero(bow[:, idx])
        print(f"  {word:20s}  idf={idf[idx]:.4f}  appears in {doc_freq} chapter(s)")

    # Most common words (lowest IDF) - shows what TF-IDF downweights
    print("\n" + "=" * 60)
    print("TOP 10 MOST COMMON WORDS (lowest IDF, downweighted)")
    print("=" * 60)
    bottom_idf = np.argsort(idf)[:10]
    for idx in bottom_idf:
        word = idx_to_word[idx]
        doc_freq = np.count_nonzero(bow[:, idx])
        total_count = int(bow[:, idx].sum())
        print(f"  {word:15s}  idf={idf[idx]:.4f}  in {doc_freq} chapters  total={total_count}")

    # Chapter similarity using TF-IDF (vs raw BoW)
    print("\n" + "=" * 60)
    print("MOST SIMILAR CHAPTER PAIRS (TF-IDF cosine)")
    print("=" * 60)
    n = len(names)
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(tfidf[i], tfidf[j])
            similarities.append((sim, i, j))
    similarities.sort(reverse=True)

    for sim, i, j in similarities[:10]:
        print(f"  {sim:.4f}  {names[i]}")
        print(f"           {names[j]}")
        print()

    # Least similar pairs
    print("LEAST SIMILAR CHAPTER PAIRS (TF-IDF cosine)")
    print("=" * 60)
    for sim, i, j in similarities[-5:]:
        print(f"  {sim:.4f}  {names[i]}")
        print(f"           {names[j]}")
        print()


if __name__ == "__main__":
    main()
