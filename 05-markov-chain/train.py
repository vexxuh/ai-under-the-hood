"""Train a variable-order Markov chain on The War of the Worlds."""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.utils import get_all_tokens

MODEL_PATH = Path(__file__).parent / "markov_model.pkl"
DEFAULT_ORDER = 4


def train(order: int = DEFAULT_ORDER):
    print(f"Training Markov chain (order={order})...")
    tokens = get_all_tokens()
    print(f"Total tokens: {len(tokens):,}")

    # Build chains for orders 1 through `order`
    chains: dict[int, dict[tuple, dict[str, int]]] = {}

    for n in range(1, order + 1):
        print(f"  Building order-{n} chain...")
        chain: dict[tuple, dict[str, int]] = {}
        for i in range(len(tokens) - n):
            context = tuple(tokens[i:i + n])
            next_word = tokens[i + n]
            if context not in chain:
                chain[context] = {}
            chain[context][next_word] = chain[context].get(next_word, 0) + 1
        chains[n] = chain
        print(f"    Unique contexts: {len(chain):,}")

    model = {
        "chains": chains,
        "max_order": order,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"\nModel saved to {MODEL_PATH}")

    # Example
    chain = chains[order]
    example_keys = list(chain.keys())[:3]
    for key in example_keys:
        top = sorted(chain[key].items(), key=lambda x: -x[1])[:3]
        context = " ".join(key)
        followers = ", ".join(f"{w}({c})" for w, c in top)
        print(f"  '{context}' → {followers}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, default=DEFAULT_ORDER)
    args = parser.parse_args()
    train(order=args.order)
