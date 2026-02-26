"""
Spatial Context Networks (SCN)
Geometric Semantic Routing in Neural Architectures

Author: Furkan Nar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GeometricActivation(nn.Module):
    """
    Geometric activation function based on normalized Euclidean distance.
    
    Each neuron acts as a point-mass with a learnable centroid in d-dimensional space.
    Activation is inversely proportional to the normalized distance from the centroid:
    
        f(v) = 1 / (||v - mu||_2 / sqrt(d) + epsilon)
    
    Args:
        n_neurons (int): Number of neurons (centroids) in this layer.
        dim (int): Dimensionality of the input semantic space.
        stability_factor (float): SF in the paper; epsilon = 1/SF. Default: 10.0
    """

    def __init__(self, n_neurons: int, dim: int, stability_factor: float = 10.0):
        super().__init__()
        self.n_neurons = n_neurons
        self.dim = dim
        self.epsilon = 1.0 / stability_factor

        # Learnable centroids: shape (n_neurons, dim)
        self.centroids = nn.Parameter(torch.randn(n_neurons, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, dim)
        Returns:
            activations: Tensor of shape (batch_size, n_neurons)
        """
        # x: (B, dim) -> (B, 1, dim)
        # centroids: (n_neurons, dim) -> (1, n_neurons, dim)
        diff = x.unsqueeze(1) - self.centroids.unsqueeze(0)  # (B, n_neurons, dim)
        dist = torch.norm(diff, dim=-1)                       # (B, n_neurons)
        normalized_dist = dist / math.sqrt(self.dim)
        activations = 1.0 / (normalized_dist + self.epsilon)
        return activations


class SemanticRoutingLayer(nn.Module):
    """
    Semantic routing layer that selectively activates neurons based on
    geometric affinity to the input.

    Active set: S = { n_i | f_i(q) > tau }
    Binary mask: M_ij = I[ f_j(v_i) > tau ]

    Args:
        n_neurons (int): Number of neurons.
        dim (int): Input dimensionality.
        routing_threshold (float): Activation threshold tau. Default: 0.5
        stability_factor (float): Passed to GeometricActivation. Default: 10.0
    """

    def __init__(
        self,
        n_neurons: int,
        dim: int,
        routing_threshold: float = 0.5,
        stability_factor: float = 10.0,
    ):
        super().__init__()
        self.routing_threshold = routing_threshold
        self.geo_activation = GeometricActivation(n_neurons, dim, stability_factor)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor of shape (batch_size, dim)
        Returns:
            activations: Raw activations, shape (batch_size, n_neurons)
            mask: Binary routing mask, shape (batch_size, n_neurons)
        """
        activations = self.geo_activation(x)
        mask = (activations > self.routing_threshold).float()
        return activations, mask


class ConnectionDensityLayer(nn.Module):
    """
    Connection density weighting with adaptive scaling and explosion control.

    C = sum_{i in S} w_i / (alpha / z)

    where alpha = total neurons, z = |S| (active neurons).
    When C > tau_exp, square-root damping is applied: C_stable = sqrt(C).

    Args:
        n_neurons (int): Total number of neurons (alpha).
        explosion_threshold (float): tau_exp. Default: 2.0
    """

    def __init__(self, n_neurons: int, explosion_threshold: float = 2.0):
        super().__init__()
        self.n_neurons = n_neurons
        self.explosion_threshold = explosion_threshold

        # Learnable per-neuron connection weights
        self.connection_weights = nn.Parameter(torch.randn(n_neurons))

    def forward(self, activations: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            activations: Shape (batch_size, n_neurons)
            mask: Binary mask, shape (batch_size, n_neurons)
        Returns:
            context: Scalar context score per sample, shape (batch_size, 1)
        """
        z = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, 1)
        alpha = float(self.n_neurons)

        # Weighted masked activations
        weighted = activations * mask * self.connection_weights.unsqueeze(0)  # (B, n)
        context = weighted.sum(dim=-1, keepdim=True) / (alpha / z)            # (B, 1)

        # Explosion control: sqrt damping
        context = torch.where(
            context > self.explosion_threshold,
            torch.sqrt(context.abs() + 1e-8) * context.sign(),
            context,
        )
        return context


class SpatialContextNetwork(nn.Module):
    """
    Spatial Context Network (SCN).

    Full architecture:
        1. SemanticRoutingLayer  — geometric activation + binary routing mask
        2. ConnectionDensityLayer — adaptive normalization + explosion control
        3. Linear projection     — map context score to output space
        4. Pattern distribution  — element-wise multiply by softmax(pattern_weights)

    Args:
        input_dim (int): Dimensionality of input features.
        n_neurons (int): Number of hidden geometric neurons. Default: 32
        output_dim (int): Number of output classes/dimensions. Default: 4
        routing_threshold (float): Routing threshold tau. Default: 0.5
        stability_factor (float): Controls epsilon = 1/SF. Default: 10.0
        explosion_threshold (float): Threshold for sqrt damping. Default: 2.0

    Example::

        model = SpatialContextNetwork(input_dim=10, n_neurons=32, output_dim=4)
        x = torch.randn(8, 10)
        output = model(x)  # (8, 4)
    """

    def __init__(
        self,
        input_dim: int = 10,
        n_neurons: int = 32,
        output_dim: int = 4,
        routing_threshold: float = 0.5,
        stability_factor: float = 10.0,
        explosion_threshold: float = 2.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_neurons = n_neurons
        self.output_dim = output_dim

        self.routing = SemanticRoutingLayer(
            n_neurons, input_dim, routing_threshold, stability_factor
        )
        self.density = ConnectionDensityLayer(n_neurons, explosion_threshold)
        self.projection = nn.Linear(1, output_dim)

        # Pattern prior weights (learnable)
        self.pattern_weights = nn.Parameter(torch.zeros(output_dim))

        # Initialise pattern weights to approximate the priors from the paper
        # [Mathematics=0.38, Language=0.25, Vision=0.22, Reasoning=0.15]
        with torch.no_grad():
            prior = torch.tensor([0.38, 0.25, 0.22, 0.15])
            if output_dim == 4:
                self.pattern_weights.copy_(torch.log(prior + 1e-8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            output: Tensor of shape (batch_size, output_dim)
        """
        activations, mask = self.routing(x)
        context = self.density(activations, mask)
        hidden = self.projection(context)                   # (B, output_dim)
        output = hidden * F.softmax(self.pattern_weights, dim=-1)
        return output

    def get_network_stats(self, x: torch.Tensor) -> dict:
        """
        Returns diagnostic statistics for a batch of inputs.

        Returns:
            dict with keys: mean_active_neurons, network_efficiency,
                            mean_context_score, activations, mask
        """
        with torch.no_grad():
            activations, mask = self.routing(x)
            context = self.density(activations, mask)
            active = mask.sum(dim=-1)
            return {
                "mean_active_neurons": active.mean().item(),
                "network_efficiency": (active / self.n_neurons).mean().item(),
                "mean_context_score": context.mean().item(),
                "activations": activations,
                "mask": mask,
            }
