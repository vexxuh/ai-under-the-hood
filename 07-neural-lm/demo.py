"""Generate text using the trained neural language model."""

import pickle
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.utils import tokenize

MODEL_PATH = Path(__file__).parent / "neural_lm.pt"
VOCAB_PATH = Path(__file__).parent / "neural_lm_vocab.pkl"


class FeedforwardLM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = torch.nn.Linear(context_size * embedding_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        embeds = embeds.view(embeds.size(0), -1)
        h = self.relu(self.fc1(embeds))
        return self.fc2(h)


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = FeedforwardLM(
        checkpoint["vocab_size"],
        checkpoint["embedding_dim"],
        checkpoint["context_size"],
        checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(VOCAB_PATH, "rb") as f:
        vocab_data = pickle.load(f)

    return model, vocab_data, checkpoint["context_size"]


def generate(model, vocab_data, context_size, seed_text=None, length=50, temperature=0.8):
    """Generate text by feeding context through the model and sampling."""
    vocab = vocab_data["vocab"]
    idx_to_word = vocab_data["idx_to_word"]
    unk_id = vocab.get("<UNK>", 0)

    if seed_text:
        tokens = tokenize(seed_text)
        token_ids = [vocab.get(t, unk_id) for t in tokens]
    else:
        # Use a common starting context
        seed = ["the", "martians", "had", "come", "to"]
        token_ids = [vocab.get(t, unk_id) for t in seed]

    # Ensure we have enough context
    while len(token_ids) < context_size:
        token_ids.insert(0, unk_id)

    generated_ids = list(token_ids)

    with torch.no_grad():
        for _ in range(length):
            context = torch.tensor([generated_ids[-context_size:]], dtype=torch.long)
            logits = model(context)
            # Apply temperature
            logits = logits / temperature
            probs = F.softmax(logits[0], dim=0)
            # Sample from distribution
            next_id = torch.multinomial(probs, 1).item()
            generated_ids.append(next_id)

    # Convert back to words
    words = [idx_to_word.get(i, "<UNK>") for i in generated_ids]
    return " ".join(words)


def main():
    print("Loading neural language model...")
    model, vocab_data, context_size = load_model()
    print(f"Vocabulary size: {len(vocab_data['vocab']):,}")
    print(f"Context size: {context_size}")

    print("\n" + "=" * 60)
    print("NEURAL LM TEXT GENERATION")
    print("=" * 60)

    seeds = [
        "the martians had come to",
        "in the darkness of the",
        "i saw the heat ray",
        "the people of london",
    ]

    for seed in seeds:
        print(f"\nSeed: \"{seed}\"")
        for temp in [0.5, 0.8, 1.2]:
            text = generate(model, vocab_data, context_size,
                          seed_text=seed, length=30, temperature=temp)
            print(f"  [temp={temp}] {text}")

    print("\n" + "=" * 60)
    print("EFFECT OF TEMPERATURE")
    print("=" * 60)
    seed = "the war of the worlds"
    for temp in [0.3, 0.5, 0.8, 1.0, 1.5]:
        text = generate(model, vocab_data, context_size,
                      seed_text=seed, length=40, temperature=temp)
        print(f"\n[temp={temp}]")
        print(f"  {text}")


if __name__ == "__main__":
    main()
