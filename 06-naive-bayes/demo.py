"""Demo: classify text snippets as first-half vs second-half style."""

import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.utils import tokenize, clean_tokens

MODEL_PATH = Path(__file__).parent / "naive_bayes_model.pkl"


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def classify(text: str, model) -> tuple[int, dict[int, float]]:
    """Classify text, returning predicted class and log-probabilities."""
    tokens = clean_tokens(tokenize(text), remove_punctuation=True)
    vocab = model["vocab"]
    scores = {}
    for c in [0, 1]:
        score = model["log_priors"][c]
        for t in tokens:
            if t in vocab:
                score += model["log_likelihoods"][c][vocab[t]]
        scores[c] = score

    # Normalize to approximate probabilities
    max_score = max(scores.values())
    exp_scores = {c: math.exp(s - max_score) for c, s in scores.items()}
    total = sum(exp_scores.values())
    probs = {c: exp_scores[c] / total for c in exp_scores}

    predicted = max(scores, key=scores.get)
    return predicted, probs


def main():
    print("Loading Naive Bayes model...")
    model = load_model()
    label_names = model["label_names"]
    print(f"Vocabulary size: {model['vocab_size']}")
    print(f"Classes: {label_names}")

    print("\n" + "=" * 60)
    print("CLASSIFYING TEXT SNIPPETS")
    print("=" * 60)

    snippets = [
        # Should lean first-half (arrival, discovery, astronomy)
        "The planet Mars was observed through the telescope, and strange flashes of light were seen on its surface.",
        "The cylinder fell from the sky and landed on Horsell Common, creating a great pit in the sandy ground.",
        "The military brought up guns and artillery to surround the pit where the cylinder had fallen.",

        # Should lean second-half (survival, destruction, aftermath)
        "The red weed grew everywhere, choking the rivers and covering the ruins of the destroyed buildings.",
        "I hid in the ruined house for days, starving and afraid, listening to the Martians outside.",
        "The Martians were dead, killed by bacteria and disease, their machines standing silent and still.",

        # Ambiguous
        "The heat ray swept across the land, destroying everything in its path with terrible fire.",
        "People fled in terror through the dark streets as the invaders advanced.",
    ]

    for snippet in snippets:
        pred, probs = classify(snippet, model)
        print(f"\n\"{snippet[:80]}...\"" if len(snippet) > 80 else f"\n\"{snippet}\"")
        print(f"  Predicted: {label_names[pred]}")
        for c in [0, 1]:
            print(f"    {label_names[c]}: {probs[c]:.4f}")


if __name__ == "__main__":
    main()
