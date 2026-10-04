"""
KAN-ProtoNet: Kolmogorov-Arnold Network as Metric Head for Few-Shot Learning

Three formulations:
  - Option A: KAN as embedding transform (prototypes computed in KAN-transformed space)
  - Option B: KAN as learned distance function (per-dimension non-linear distance)
  - Option C: Dual KAN (both transform + distance) — ablation upper-bound

Reference architecture:
  ResNet18 → 512-dim embedding → KAN transform/distance → prototype-based classification

Integration:
  This module follows the FHIST FSmethod interface so it can be used as a drop-in
  replacement for ProtoNet in the FHIST codebase.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from typing import Tuple, Optional, Literal

import sys
import os

# Add parent path for imports when used standalone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kan import KANLinear, KANHead

# Import FHIST utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'fhist'))
from src.methods.utils import get_one_hot, compute_centroids, extract_features
from src.methods.method import FSmethod


class KANProtoNet(FSmethod):
    """
    KAN-ProtoNet: Prototypical Network with KAN-based metric head.

    Supports three formulations (set via `kan_mode`):

    Option A ('transform'):
        Prototypes are computed in KAN-transformed feature space.
        g_φ transforms both support and query embeddings before
        computing Euclidean distance to prototypes.

    Option B ('distance'):
        Prototypes are computed in original embedding space.
        KAN learns a per-dimension non-linear distance function
        that replaces Euclidean distance.

    Option C ('dual'):
        Both KAN transform and KAN distance are applied.
        Most expressive but highest overfitting risk.

    Args:
        args: Namespace with configuration (from FHIST config system)
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # KAN configuration (with defaults)
        self.kan_mode: str = getattr(args, 'kan_mode', 'transform')  # 'transform', 'distance', 'dual'
        self.feat_dim: int = getattr(args, 'feat_dim', 512)  # ResNet18 output dim
        self.kan_hidden: Optional[int] = getattr(args, 'kan_hidden', None)  # None = single layer KAN
        self.kan_out_dim: int = getattr(args, 'kan_out_dim', 512)  # Output dim for transform
        self.grid_size: int = getattr(args, 'kan_grid_size', 5)
        self.spline_order: int = getattr(args, 'kan_spline_order', 3)
        self.kan_dropout: float = getattr(args, 'kan_dropout', 0.0)
        self.temperature: float = getattr(args, 'kan_temperature', 1.0)

        # Build KAN components based on mode
        if self.kan_mode in ('transform', 'dual'):
            self.kan_transform = KANHead(
                in_features=self.feat_dim,
                out_features=self.kan_out_dim,
                hidden_features=self.kan_hidden,
                grid_size=self.grid_size,
                spline_order=self.spline_order,
                dropout=self.kan_dropout,
            )
        else:
            self.kan_transform = None

        if self.kan_mode in ('distance', 'dual'):
            # For distance mode: KAN learns a per-dimension scoring function
            # Input = difference per dimension, output = scalar contribution
            # We use a single KANLinear layer with in=feat_dim, out=1
            # This computes s(q,k) = Σ_j φ_j(q_j - c_{k,j})
            dist_in = self.kan_out_dim if self.kan_mode == 'dual' else self.feat_dim
            self.kan_distance = KANLinear(
                in_features=dist_in,
                out_features=1,
                grid_size=self.grid_size,
                spline_order=self.spline_order,
            )
        else:
            self.kan_distance = None

    def forward(
        self,
        x_s: torch.Tensor,
        x_q: torch.Tensor,
        y_s: torch.Tensor,
        y_q: torch.Tensor,
        model: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for episodic few-shot evaluation.

        Args:
            x_s: Support images [batch, s_shot, C, H, W]
            x_q: Query images [batch, q_shot, C, H, W]
            y_s: Support labels [batch, s_shot]
            y_q: Query labels [batch, q_shot]
            model: Backbone feature extractor (ResNet18)

        Returns:
            loss: Cross-entropy loss [batch, q_shot]
            preds_q: Predicted labels [batch, q_shot]
        """
        num_classes = y_s.unique().size(0)

        # Extract features from backbone
        if not self.training:
            with torch.no_grad():
                z_s = extract_features(x_s, model)  # [batch, s_shot, 512]
                z_q = extract_features(x_q, model)  # [batch, q_shot, 512]
        else:
            z_s = extract_features(x_s, model)
            z_q = extract_features(x_q, model)

        # Apply KAN-based metric computation
        if self.kan_mode == 'transform':
            log_probas = self._forward_transform(z_s, z_q, y_s, num_classes)
        elif self.kan_mode == 'distance':
            log_probas = self._forward_distance(z_s, z_q, y_s, num_classes)
        elif self.kan_mode == 'dual':
            log_probas = self._forward_dual(z_s, z_q, y_s, num_classes)
        else:
            raise ValueError(f"Unknown kan_mode: {self.kan_mode}")

        # Compute loss and predictions
        one_hot_q = get_one_hot(y_q, num_classes)
        ce = -(one_hot_q * log_probas).sum(-1)  # [batch, q_shot]
        preds_q = log_probas.detach().exp().argmax(2)  # [batch, q_shot]

        return ce, preds_q

    def _forward_transform(
        self, z_s: torch.Tensor, z_q: torch.Tensor,
        y_s: torch.Tensor, num_classes: int
    ) -> torch.Tensor:
        """
        Option A: KAN embedding transform.

        c_k = (1/|S_k|) Σ g_φ(f_θ(x_i))
        p(y=k|x_q) ∝ exp(-d(g_φ(f_θ(x_q)), c_k))
        """
        # Transform support and query through KAN
        z_s_kan = self.kan_transform(z_s)  # [batch, s_shot, kan_out_dim]
        z_q_kan = self.kan_transform(z_q)  # [batch, q_shot, kan_out_dim]

        # Compute prototypes in KAN-transformed space
        centroids = compute_centroids(z_s_kan, y_s)  # [batch, num_class, kan_out_dim]

        # L2 distance in transformed space
        l2_distance = (
            -2 * z_q_kan.matmul(centroids.transpose(1, 2))
            + (centroids ** 2).sum(2).unsqueeze(1)
            + (z_q_kan ** 2).sum(2).unsqueeze(-1)
        )  # [batch, q_shot, num_class]

        log_probas = (-l2_distance / self.temperature).log_softmax(-1)
        return log_probas

    def _forward_distance(
        self, z_s: torch.Tensor, z_q: torch.Tensor,
        y_s: torch.Tensor, num_classes: int
    ) -> torch.Tensor:
        """
        Option B: KAN learned distance function.

        c_k = (1/|S_k|) Σ f_θ(x_i)
        s(q,k) = Σ_j φ_j(f_θ(x_q)_j - c_{k,j})
        p(y=k|x_q) ∝ exp(s(q,k))
        """
        # Prototypes in original embedding space
        centroids = compute_centroids(z_s, y_s)  # [batch, num_class, feat_dim]

        batch_size = z_q.size(0)
        q_shot = z_q.size(1)

        # Compute per-dimension differences: z_q - c_k for all (q, k) pairs
        # z_q: [batch, q, d], centroids: [batch, K, d]
        # diff: [batch, q, K, d]
        diff = z_q.unsqueeze(2) - centroids.unsqueeze(1)

        # Reshape for KAN: [batch * q * K, d]
        diff_flat = diff.reshape(-1, self.feat_dim)

        # KAN distance: maps per-dimension differences to scalar score
        scores_flat = self.kan_distance(diff_flat)  # [batch*q*K, 1]
        scores = scores_flat.reshape(batch_size, q_shot, num_classes)  # [batch, q, K]

        log_probas = (scores / self.temperature).log_softmax(-1)
        return log_probas

    def _forward_dual(
        self, z_s: torch.Tensor, z_q: torch.Tensor,
        y_s: torch.Tensor, num_classes: int
    ) -> torch.Tensor:
        """
        Option C: Dual KAN (transform + distance).

        c_k = (1/|S_k|) Σ g_φ(f_θ(x_i))
        s(q,k) = Σ_j ψ_j(g_φ(f_θ(x_q))_j - c_{k,j})
        p(y=k|x_q) ∝ exp(s(q,k))
        """
        # Transform through KAN
        z_s_kan = self.kan_transform(z_s)  # [batch, s, kan_out_dim]
        z_q_kan = self.kan_transform(z_q)  # [batch, q, kan_out_dim]

        # Prototypes in KAN-transformed space
        centroids = compute_centroids(z_s_kan, y_s)  # [batch, K, kan_out_dim]

        batch_size = z_q_kan.size(0)
        q_shot = z_q_kan.size(1)

        # Per-dimension differences in KAN space
        diff = z_q_kan.unsqueeze(2) - centroids.unsqueeze(1)
        diff_flat = diff.reshape(-1, self.kan_out_dim)

        # KAN distance scoring
        scores_flat = self.kan_distance(diff_flat)
        num_classes = centroids.size(1)
        scores = scores_flat.reshape(batch_size, q_shot, num_classes)

        log_probas = (scores / self.temperature).log_softmax(-1)
        return log_probas

    def get_kan_parameters(self):
        """Return only KAN-specific parameters (for separate optimizer)."""
        params = []
        if self.kan_transform is not None:
            params.extend(self.kan_transform.parameters())
        if self.kan_distance is not None:
            params.extend(self.kan_distance.parameters())
        return params

    def count_kan_parameters(self) -> dict:
        """Count parameters for computational cost analysis."""
        counts = {'total': 0}
        if self.kan_transform is not None:
            n = sum(p.numel() for p in self.kan_transform.parameters())
            counts['transform'] = n
            counts['total'] += n
        if self.kan_distance is not None:
            n = sum(p.numel() for p in self.kan_distance.parameters())
            counts['distance'] = n
            counts['total'] += n
        return counts


class KANProtoNetFinetune(FSmethod):
    """
    KAN-head with simple fine-tuning (non-metric-learning variant).

    This is the simpler baseline: ResNet18 → KAN classifier head,
    trained with cross-entropy. For comparison with KAN-ProtoNet.

    Used during base training with standard (non-episodic) training,
    and evaluated episodically with nearest-centroid in KAN-output space.
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        self.feat_dim: int = getattr(args, 'feat_dim', 512)
        self.kan_hidden: Optional[int] = getattr(args, 'kan_hidden', None)
        self.grid_size: int = getattr(args, 'kan_grid_size', 5)
        self.spline_order: int = getattr(args, 'kan_spline_order', 3)

    def forward(
        self,
        x_s: torch.Tensor,
        x_q: torch.Tensor,
        y_s: torch.Tensor,
        y_q: torch.Tensor,
        model: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        At test time, use nearest-centroid in embedding space.
        KAN-head is not used at inference — only during base training.
        """
        num_classes = y_s.unique().size(0)

        if not self.training:
            with torch.no_grad():
                z_s = extract_features(x_s, model)
                z_q = extract_features(x_q, model)
        else:
            z_s = extract_features(x_s, model)
            z_q = extract_features(x_q, model)

        # Standard ProtoNet-style evaluation
        centroids = compute_centroids(z_s, y_s)

        l2_distance = (
            -2 * z_q.matmul(centroids.transpose(1, 2))
            + (centroids ** 2).sum(2).unsqueeze(1)
            + (z_q ** 2).sum(2).unsqueeze(-1)
        )

        log_probas = (-l2_distance).log_softmax(-1)
        one_hot_q = get_one_hot(y_q, num_classes)
        ce = -(one_hot_q * log_probas).sum(-1)
        preds_q = log_probas.detach().exp().argmax(2)

        return ce, preds_q
