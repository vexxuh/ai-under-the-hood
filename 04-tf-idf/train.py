"""Build TF-IDF representations for each chapter of The War of the Worlds."""

import math
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.utils import get_chapters, tokenize, clean_tokens, build_vocab

MODEL_PATH = Path(__file__).parent / "tfidf_model.pkl"


def compute_tf(bow_row: np.ndarray) -> np.ndarray:
    """Term frequency: count / total tokens in document."""
    total = bow_row.sum()
    if total == 0:
        return bow_row
    return bow_row / total


def compute_idf(bow_matrix: np.ndarray) -> np.ndarray:
    """Inverse document frequency: log(N / df) for each term.

    Uses standard IDF with +1 smoothing in the denominator to avoid
    division by zero, without the additive +1 that sklearn uses. This
    produces stronger separation between common and rare terms.
    """
    n_docs = bow_matrix.shape[0]
    # df = number of documents containing the term (count > 0)
    df = np.count_nonzero(bow_matrix, axis=0).astype(np.float32)
    # Smooth denominator only to avoid div-by-zero
    idf = np.log(n_docs / (df + 1))
    return idf.astype(np.float32)


def train():
    print("Loading chapters...")
    chapters = get_chapters()
    print(f"Found {len(chapters)} chapters")

    # Tokenize each chapter
    chapter_tokens = []
    for ch in chapters:
        tokens = clean_tokens(tokenize(ch["text"]), remove_punctuation=True)
        chapter_tokens.append(tokens)

    # Build vocabulary from all tokens
    all_tokens = [t for ct in chapter_tokens for t in ct]
    vocab = build_vocab(all_tokens, min_count=3)
    print(f"Vocabulary size (min_count=3): {len(vocab):,}")

    # Build raw count matrix: chapters x vocab_size
    bow_matrix = np.zeros((len(chapters), len(vocab)), dtype=np.float32)
    for i, tokens in enumerate(chapter_tokens):
        for t in tokens:
            if t in vocab:
                bow_matrix[i, vocab[t]] += 1

    # Compute IDF across all documents
    idf = compute_idf(bow_matrix)

    # Compute TF-IDF matrix
    tfidf_matrix = np.zeros_like(bow_matrix)
    for i in range(len(chapters)):
        tf = compute_tf(bow_matrix[i])
        tfidf_matrix[i] = tf * idf

    # Chapter metadata
    chapter_names = [
        f"Book {ch['book']}, Ch {ch['chapter']}: {ch['title']}"
        for ch in chapters
    ]

    idx_to_word = {i: w for w, i in vocab.items()}

    model = {
        "tfidf_matrix": tfidf_matrix,
        "bow_matrix": bow_matrix,
        "idf": idf,
        "vocab": vocab,
        "idx_to_word": idx_to_word,
        "chapter_names": chapter_names,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {MODEL_PATH}")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Show top TF-IDF words for the first chapter
    print(f"\nTop 10 TF-IDF words in '{chapter_names[0]}':")
    top_idx = np.argsort(-tfidf_matrix[0])[:10]
    for idx in top_idx:
        word = idx_to_word[idx]
        score = tfidf_matrix[0, idx]
        count = int(bow_matrix[0, idx])
        print(f"  {word:15s}  tfidf={score:.4f}  count={count}")

    # Compare with raw counts to show the difference
    print(f"\nTop 10 raw-count words in '{chapter_names[0]}':")
    top_raw = np.argsort(-bow_matrix[0])[:10]
    for idx in top_raw:
        word = idx_to_word[idx]
        count = int(bow_matrix[0, idx])
        score = tfidf_matrix[0, idx]
        print(f"  {word:15s}  count={count}  tfidf={score:.4f}")


if __name__ == "__main__":
    train()
