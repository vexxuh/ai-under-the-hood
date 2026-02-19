"""Train a Naive Bayes classifier: first-half vs second-half of the book."""

import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.utils import get_chapters, tokenize, clean_tokens

MODEL_PATH = Path(__file__).parent / "naive_bayes_model.pkl"


def train():
    print("Loading chapters...")
    chapters = get_chapters()
    n = len(chapters)
    mid = n // 2
    print(f"Total chapters: {n}, split at chapter {mid}")

    # Label chapters: 0 = first half, 1 = second half
    labels = [0 if i < mid else 1 for i in range(n)]
    label_names = {0: "First Half (Arrival & Discovery)", 1: "Second Half (Survival & Aftermath)"}

    # Tokenize chapters
    chapter_tokens = []
    for ch in chapters:
        tokens = clean_tokens(tokenize(ch["text"]), remove_punctuation=True)
        chapter_tokens.append(tokens)

    # Build vocabulary
    all_tokens = [t for ct in chapter_tokens for t in ct]
    vocab: dict[str, int] = {}
    freq: dict[str, int] = {}
    for t in all_tokens:
        freq[t] = freq.get(t, 0) + 1
    idx = 0
    for word in sorted(freq):
        if freq[word] >= 3:
            vocab[word] = idx
            idx += 1
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Compute class priors and word likelihoods with Laplace smoothing
    class_counts = {0: 0, 1: 0}
    word_counts: dict[int, dict[int, int]] = {0: {}, 1: {}}
    class_total_words = {0: 0, 1: 0}

    for i, tokens in enumerate(chapter_tokens):
        label = labels[i]
        class_counts[label] += 1
        for t in tokens:
            if t in vocab:
                wid = vocab[t]
                word_counts[label][wid] = word_counts[label].get(wid, 0) + 1
                class_total_words[label] += 1

    # Log priors
    total_docs = sum(class_counts.values())
    log_priors = {c: math.log(count / total_docs) for c, count in class_counts.items()}

    # Log likelihoods with Laplace smoothing (alpha=1)
    alpha = 1.0
    log_likelihoods: dict[int, dict[int, float]] = {0: {}, 1: {}}
    for c in [0, 1]:
        total = class_total_words[c] + alpha * vocab_size
        for wid in range(vocab_size):
            count = word_counts[c].get(wid, 0) + alpha
            log_likelihoods[c][wid] = math.log(count / total)

    model = {
        "log_priors": log_priors,
        "log_likelihoods": log_likelihoods,
        "vocab": vocab,
        "vocab_size": vocab_size,
        "label_names": label_names,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {MODEL_PATH}")

    # Test on training data
    print("\nTraining accuracy:")
    correct = 0
    for i, tokens in enumerate(chapter_tokens):
        true_label = labels[i]
        pred = classify(tokens, model)
        if pred == true_label:
            correct += 1
        status = "✓" if pred == true_label else "✗"
        ch = chapters[i]
        print(f"  {status} Book {ch['book']}, Ch {ch['chapter']}: "
              f"predicted={label_names[pred]}, actual={label_names[true_label]}")
    print(f"\nAccuracy: {correct}/{n} ({correct/n*100:.1f}%)")

    # Show most discriminative words
    print("\nMost indicative words for each class:")
    for c in [0, 1]:
        other = 1 - c
        ratios = []
        for wid in range(vocab_size):
            ratio = log_likelihoods[c][wid] - log_likelihoods[other][wid]
            ratios.append((ratio, wid))
        ratios.sort(reverse=True)
        idx_to_word = {i: w for w, i in vocab.items()}
        words = [idx_to_word[wid] for _, wid in ratios[:10]]
        print(f"  {label_names[c]}: {', '.join(words)}")


def classify(tokens: list[str], model) -> int:
    """Classify a list of tokens into a class."""
    vocab = model["vocab"]
    best_class = 0
    best_score = float("-inf")
    for c in [0, 1]:
        score = model["log_priors"][c]
        for t in tokens:
            if t in vocab:
                score += model["log_likelihoods"][c][vocab[t]]
        if score > best_score:
            best_score = score
            best_class = c
    return best_class


if __name__ == "__main__":
    train()
