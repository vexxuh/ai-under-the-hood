"""Train a simple feedforward neural language model with PyTorch."""

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.utils import get_all_tokens, build_vocab

MODEL_PATH = Path(__file__).parent / "neural_lm.pt"
VOCAB_PATH = Path(__file__).parent / "neural_lm_vocab.pkl"

# Hyperparameters
CONTEXT_SIZE = 5
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 256
EPOCHS = 10
LR = 0.001
MIN_WORD_COUNT = 2


class TextDataset(Dataset):
    def __init__(self, token_ids: list[int], context_size: int):
        self.data = token_ids
        self.context_size = context_size

    def __len__(self):
        return len(self.data) - self.context_size

    def __getitem__(self, idx):
        context = self.data[idx:idx + self.context_size]
        target = self.data[idx + self.context_size]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class FeedforwardLM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, context_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: (batch, context_size)
        embeds = self.embedding(x)  # (batch, context_size, embedding_dim)
        embeds = embeds.view(embeds.size(0), -1)  # (batch, context_size * embedding_dim)
        h = self.relu(self.fc1(embeds))
        logits = self.fc2(h)
        return logits


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading and tokenizing text...")
    tokens = get_all_tokens(remove_punctuation=False)
    print(f"Total tokens: {len(tokens):,}")

    vocab = build_vocab(tokens, min_count=MIN_WORD_COUNT)
    # Add <UNK> token
    unk_id = len(vocab)
    vocab["<UNK>"] = unk_id
    idx_to_word = {i: w for w, i in vocab.items()}
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size:,}")

    # Convert tokens to IDs
    token_ids = [vocab.get(t, unk_id) for t in tokens]

    # Create dataset
    dataset = TextDataset(token_ids, CONTEXT_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"Training samples: {len(dataset):,}")

    # Build model
    model = FeedforwardLM(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for context, target in dataloader:
            context = context.to(device)
            target = target.to(device)

            logits = model(context)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        perplexity = np.exp(avg_loss)
        print(f"Epoch {epoch + 1}/{EPOCHS} — Loss: {avg_loss:.4f}, Perplexity: {perplexity:.1f}")

    # Save model and vocab
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "embedding_dim": EMBEDDING_DIM,
        "context_size": CONTEXT_SIZE,
        "hidden_dim": HIDDEN_DIM,
    }, MODEL_PATH)

    with open(VOCAB_PATH, "wb") as f:
        pickle.dump({"vocab": vocab, "idx_to_word": idx_to_word}, f)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Vocab saved to {VOCAB_PATH}")


if __name__ == "__main__":
    train()
