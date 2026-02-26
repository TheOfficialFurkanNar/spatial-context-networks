"""
Training script for Spatial Context Networks (SCN).

Example usage:
    python train.py --input_dim 10 --n_neurons 32 --output_dim 4 --epochs 50
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import SpatialContextNetwork


def make_synthetic_dataset(n_samples=256, input_dim=10, output_dim=4, seed=42):
    """Creates a simple synthetic classification dataset for demonstration."""
    torch.manual_seed(seed)
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, output_dim, (n_samples,))
    return TensorDataset(X, y)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Data
    dataset = make_synthetic_dataset(
        n_samples=args.n_samples,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model
    model = SpatialContextNetwork(
        input_dim=args.input_dim,
        n_neurons=args.n_neurons,
        output_dim=args.output_dim,
        routing_threshold=args.routing_threshold,
        stability_factor=args.stability_factor,
        explosion_threshold=args.explosion_threshold,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params}")
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(X_batch)
            correct += (logits.argmax(dim=-1) == y_batch).sum().item()

        avg_loss = total_loss / len(dataset)
        accuracy = correct / len(dataset)

        if epoch % 10 == 0 or epoch == 1:
            # Network efficiency stats
            model.eval()
            with torch.no_grad():
                sample_x = torch.randn(args.batch_size, args.input_dim).to(device)
                stats = model.get_network_stats(sample_x)
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Loss: {avg_loss:.4f} | Acc: {accuracy:.3f} | "
                f"Active neurons: {stats['mean_active_neurons']:.1f}/{args.n_neurons} "
                f"(eff={stats['network_efficiency']:.2f}) | "
                f"Context score: {stats['mean_context_score']:.3f}"
            )

    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
        print(f"\nModel saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Spatial Context Network")

    # Architecture
    parser.add_argument("--input_dim", type=int, default=10)
    parser.add_argument("--n_neurons", type=int, default=32)
    parser.add_argument("--output_dim", type=int, default=4)
    parser.add_argument("--routing_threshold", type=float, default=0.5)
    parser.add_argument("--stability_factor", type=float, default=10.0)
    parser.add_argument("--explosion_threshold", type=float, default=2.0)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--save_path", type=str, default=None)

    args = parser.parse_args()
    train(args)
