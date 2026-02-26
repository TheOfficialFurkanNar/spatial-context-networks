# Spatial Context Networks (SCN)

> **Geometric Semantic Routing in Neural Architectures**  
> Furkan Nar — Independent Researcher  
> February 2026

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18599303.svg)](https://doi.org/10.5281/zenodo.18599303)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)

---

## Overview

Spatial Context Networks (SCN) is a novel neural architecture that treats neurons as **geometric entities in a learned semantic space**. Rather than relying on weighted linear combinations, each neuron operates as a point-mass with a learnable centroid — activating based on its distance to the input in that space.

This repository contains the reference PyTorch implementation accompanying the paper.

📄 **Paper:** [https://doi.org/10.5281/zenodo.18599303](https://doi.org/10.5281/zenodo.18599303)

### Key Ideas

- **Geometric Activation** — activation inversely proportional to normalized Euclidean distance from a learnable centroid
- **Semantic Routing** — binary hard-routing that only activates neurons geometrically close to the input
- **Connection Density Weighting** — adaptive normalization that stabilizes signal magnitude across sparsity regimes
- **Pattern Distribution** — a Bayesian prior over output patterns via learnable softmax weights

---

## Architecture

```
Input x ∈ ℝ^d
     │
     ▼
┌─────────────────────────┐
│  Semantic Routing Layer  │  ← Geometric activations + binary mask
│  f(v) = 1 / (‖v−μ‖/√d + ε)│
└─────────────────────────┘
     │ activations, mask
     ▼
┌─────────────────────────┐
│ Connection Density Layer │  ← Adaptive normalization + explosion control
│  C = Σ w_i / (α/z)      │
└─────────────────────────┘
     │ context score
     ▼
┌─────────────────────────┐
│   Linear Projection      │
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│  Pattern Distribution    │  ← h ⊙ softmax(w_p)
└─────────────────────────┘
     │
     ▼
Output o ∈ ℝ^dout
```

---

## Installation

```bash
git clone https://github.com/TheOfficialFurkanNar/spatial-context-networks.git
cd spatial-context-networks
pip install -e .
```
---

## Quick Start

```python
import torch
from model.py import SpatialContextNetwork

# Instantiate the model
model = SpatialContextNetwork(
    input_dim=10,
    n_neurons=32,
    output_dim=4,
    routing_threshold=0.5,
    stability_factor=10.0,
    explosion_threshold=2.0,
)

# Forward pass
x = torch.randn(8, 10)
output = model(x)          # shape: (8, 4)

# Diagnostic stats
stats = model.get_network_stats(x)
print(f"Network efficiency: {stats['network_efficiency']:.1%}")
print(f"Mean active neurons: {stats['mean_active_neurons']:.1f} / 32")
```

---

## Training

```bash
python train.py \
    --input_dim 10 \
    --n_neurons 32 \
    --output_dim 4 \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-3 \
    --save_path scn_model.pt
```

---

## Inference

```bash
python inference.py --checkpoint scn_model.pt --batch_size 8
```

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_dim` | 10 | Input feature dimensionality |
| `n_neurons` | 32 | Number of geometric hidden neurons |
| `output_dim` | 4 | Output dimensionality |
| `routing_threshold` τ | 0.5 | Minimum activation to route through a neuron |
| `stability_factor` SF | 10.0 | ε = 1/SF; prevents division by zero at centroid |
| `explosion_threshold` τ_exp | 2.0 | Context scores above this get √ damped |

---

## Results (Proof-of-Concept)

| Metric | Value |
|--------|-------|
| Mean active neurons | 29.1 / 32 |
| Network efficiency | 91% |
| Mean context score | 0.444 |
| Total parameters | ~500 |
| Hardware | Consumer gaming laptop (RTX) |

---

## Citation

If you use this work, please cite:

```bibtex
@article{nar2026scn,
  title   = {Spatial Context Networks: Geometric Semantic Routing in Neural Architectures},
  author  = {Nar, Furkan},
  year    = {2026},
  month   = {February},
  doi     = {10.5281/zenodo.18599303},
  url     = {https://doi.org/10.5281/zenodo.18599303},
  note    = {Independent research. Published on Zenodo and Academia.edu}
}
```

---

## License

[MIT](LICENSE) © 2026 Furkan Nar
