"""Generate text using the trained Markov chain model."""

import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_PATH = Path(__file__).parent / "markov_model.pkl"


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def sample_from(dist: dict[str, int]) -> str:
    words = list(dist.keys())
    weights = list(dist.values())
    return random.choices(words, weights=weights, k=1)[0]


def generate(model, seed=None, length=80, order=None):
    """Generate text with backoff: try highest order first, fall back to lower."""
    chains = model["chains"]
    max_order = model["max_order"] if order is None else min(order, model["max_order"])

    # Pick a starting context from the highest-order chain
    top_chain = chains[max_order]
    if seed:
        context = list(seed)
    else:
        context = list(random.choice(list(top_chain.keys())))

    words = list(context)

    for _ in range(length - len(context)):
        generated = False
        # Try from highest order down to 1
        for n in range(min(max_order, len(words)), 0, -1):
            key = tuple(words[-n:])
            if n in chains and key in chains[n]:
                next_word = sample_from(chains[n][key])
                words.append(next_word)
                generated = True
                break
        if not generated:
            # Random restart
            context = random.choice(list(top_chain.keys()))
            words.append(context[0])

    return " ".join(words)


def main():
    print("Loading Markov chain model...")
    model = load_model()
    max_order = model["max_order"]
    print(f"Max order: {max_order}")
    for n, chain in model["chains"].items():
        print(f"  Order {n}: {len(chain):,} unique contexts")

    print("\n" + "=" * 60)
    print("TEXT GENERATION (varying orders)")
    print("=" * 60)

    for order in range(1, max_order + 1):
        print(f"\n--- Order {order} ---")
        print(generate(model, length=60, order=order))

    print("\n" + "=" * 60)
    print("SEEDED GENERATION (max order)")
    print("=" * 60)
    seeds = [
        ("the", "martians"),
        ("i", "saw", "the"),
        ("in", "the", "darkness"),
    ]
    for seed in seeds:
        top_chain = model["chains"][min(len(seed), max_order)]
        if tuple(seed[-min(len(seed), max_order):]) in top_chain:
            print(f"\nSeed: '{' '.join(seed)}'")
            print(generate(model, seed=seed, length=60))


if __name__ == "__main__":
    main()
