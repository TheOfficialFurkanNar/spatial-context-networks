"""
Inference / demo script for Spatial Context Networks (SCN).
Designed for use as a HuggingFace Space or standalone demo.

Usage:
    python inference.py
    python inference.py --checkpoint path/to/model.pt --input_dim 10
"""

import argparse
import torch
import json
from spatial_context_networks import SpatialContextNetwork


PATTERN_LABELS = ["Mathematics", "Language", "Vision", "Reasoning"]


def load_model(checkpoint_path: str | None, input_dim: int, n_neurons: int, output_dim: int):
    model = SpatialContextNetwork(
        input_dim=input_dim,
        n_neurons=n_neurons,
        output_dim=output_dim,
    )
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint provided — using randomly initialized weights.")
    model.eval()
    return model


def run_inference(model: SpatialContextNetwork, x: torch.Tensor) -> dict:
    """
    Run a single forward pass and return rich diagnostic output.

    Returns:
        dict with output logits, predicted pattern, network efficiency stats.
    """
    with torch.no_grad():
        output = model(x)
        stats = model.get_network_stats(x)

    probs = torch.softmax(output, dim=-1)
    predicted_idx = probs.argmax(dim=-1)

    results = {
        "output_logits": output.tolist(),
        "output_probabilities": probs.tolist(),
        "predicted_pattern": [PATTERN_LABELS[i] for i in predicted_idx.tolist()],
        "mean_active_neurons": round(stats["mean_active_neurons"], 2),
        "network_efficiency": round(stats["network_efficiency"], 4),
        "mean_context_score": round(stats["mean_context_score"], 4),
    }
    return results


def demo(args):
    model = load_model(args.checkpoint, args.input_dim, args.n_neurons, args.output_dim)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print("  Spatial Context Network — Inference Demo")
    print(f"{'='*60}")
    print(f"  Input dim   : {args.input_dim}")
    print(f"  Hidden neurons: {args.n_neurons}")
    print(f"  Output dim  : {args.output_dim}")
    print(f"  Parameters  : {total_params}")
    print(f"{'='*60}\n")

    torch.manual_seed(42)
    x = torch.randn(args.batch_size, args.input_dim)
    print(f"Running inference on {args.batch_size} random samples...\n")

    results = run_inference(model, x)

    for i in range(args.batch_size):
        probs = results["output_probabilities"][i]
        predicted = results["predicted_pattern"][i]
        prob_str = " | ".join(
            f"{label}: {p:.3f}" for label, p in zip(PATTERN_LABELS, probs)
        )
        print(f"  Sample {i}: [{prob_str}]  → Predicted: {predicted}")

    print(f"\n  Network Stats:")
    print(f"    Active neurons : {results['mean_active_neurons']} / {args.n_neurons}")
    print(f"    Efficiency     : {results['network_efficiency']:.1%}")
    print(f"    Context score  : {results['mean_context_score']:.4f}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCN Inference Demo")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--input_dim", type=int, default=10)
    parser.add_argument("--n_neurons", type=int, default=32)
    parser.add_argument("--output_dim", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()
    demo(args)
