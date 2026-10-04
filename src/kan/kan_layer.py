"""
KAN Layer Implementation for KAN-Fine++
Efficient B-Spline based Kolmogorov-Arnold Network layer.

This is a self-contained implementation optimized for few-shot learning heads.
It does NOT require the pykan library (but is compatible with it).

Reference: Liu et al., "KAN: Kolmogorov-Arnold Networks", ICLR 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class BSplineBasis(nn.Module):
    """
    Computes B-spline basis functions for KAN edges.
    Uses efficient vectorized computation.
    """

    def __init__(self, grid_size: int = 5, spline_order: int = 3,
                 grid_range: Tuple[float, float] = (-1.0, 1.0)):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range

        # Create extended knot vector (grid_size + 2 * spline_order + 1 knots)
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1
        )
        self.register_buffer('grid', grid)

    @property
    def num_basis(self) -> int:
        """Number of B-spline basis functions."""
        return self.grid_size + self.spline_order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate B-spline basis functions at points x.

        Args:
            x: Input tensor of any shape

        Returns:
            Basis values of shape (*x.shape, num_basis)
        """
        x_shape = x.shape
        x = x.reshape(-1, 1)  # [N, 1]
        grid = self.grid  # [G]

        # Cox-de Boor recursion for B-splines
        # Order 0: piecewise constant
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()  # [N, G-1]

        # Recursion for higher orders
        for k in range(1, self.spline_order + 1):
            left_num = x - grid[:-(k + 1)]  # [N, G-k-1]
            left_den = grid[k:-1] - grid[:-(k + 1)]  # [G-k-1]
            left = left_num / (left_den + 1e-8) * bases[:, :-1]

            right_num = grid[k + 1:] - x  # [N, G-k-1]
            right_den = grid[k + 1:] - grid[1:-k]  # [G-k-1]
            right = right_num / (right_den + 1e-8) * bases[:, 1:]

            bases = left + right

        return bases.reshape(*x_shape, self.num_basis)


class KANLinear(nn.Module):
    """
    KAN Linear layer: replaces fixed linear weights with learnable
    univariate spline functions on each edge.

    For input dimension in_features and output dimension out_features,
    there are in_features * out_features edges, each parameterized
    by a B-spline with (grid_size + spline_order) coefficients.

    Additionally includes a residual linear path (SiLU activation + linear)
    as in the original KAN paper for training stability.

    Args:
        in_features:  Input dimension
        out_features: Output dimension
        grid_size:    Number of grid intervals for B-splines
        spline_order: Order of B-splines (3 = cubic)
        grid_range:   Range of the spline grid
        residual_weight: Weight for residual (base) path. 0 = pure KAN.
        grid_eps:     Small epsilon for grid initialization
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        residual_weight: float = 1.0,
        grid_eps: float = 0.02,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.residual_weight = residual_weight

        # B-spline basis
        self.basis = BSplineBasis(grid_size, spline_order, grid_range)
        num_basis = self.basis.num_basis

        # Spline coefficients: [out_features, in_features, num_basis]
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, num_basis)
        )

        # Residual (base) linear path
        if residual_weight > 0:
            self.base_weight = nn.Parameter(
                torch.empty(out_features, in_features)
            )
        else:
            self.base_weight = None

        # Optional per-edge scale
        self.spline_scaler = nn.Parameter(
            torch.ones(out_features, in_features)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with proper scaling."""
        # Spline weights: small random init
        nn.init.trunc_normal_(self.spline_weight, std=0.1)

        # Base weights: Kaiming init
        if self.base_weight is not None:
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        # Spline scaler: ones
        nn.init.ones_(self.spline_scaler)

    @property
    def num_parameters(self) -> int:
        """Total number of learnable parameters in this layer."""
        total = self.spline_weight.numel() + self.spline_scaler.numel()
        if self.base_weight is not None:
            total += self.base_weight.numel()
        return total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch, in_features] or
               [batch, seq, in_features]

        Returns:
            Output tensor of shape [batch, out_features] or
            [batch, seq, out_features]
        """
        original_shape = x.shape
        if x.dim() == 3:
            batch, seq, d = x.shape
            x = x.reshape(batch * seq, d)
        elif x.dim() > 3:
            # Handle arbitrary leading dimensions
            leading = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
        
        # --- Spline path ---
        # x: [N, in_features]
        bases = self.basis(x)  # [N, in_features, num_basis]

        # Compute spline output: sum over basis functions per edge
        # spline_weight: [out, in, basis]
        # bases: [N, in, basis]
        # result: [N, out]  (sum over in and basis)
        spline_out = torch.einsum(
            'oib,nib->no',
            self.spline_weight * self.spline_scaler.unsqueeze(-1),
            bases
        )

        # --- Base (residual) path ---
        if self.base_weight is not None:
            base_out = F.silu(x) @ self.base_weight.t()
            output = spline_out + self.residual_weight * base_out
        else:
            output = spline_out

        # Reshape back
        if len(original_shape) == 3:
            output = output.reshape(batch, seq, -1)
        elif len(original_shape) > 3:
            output = output.reshape(*leading, -1)

        return output

    def get_spline_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-edge spline importance (magnitude) for SFA attribution.

        Args:
            x: Input tensor [batch, in_features]

        Returns:
            Importance scores [out_features, in_features] — mean absolute
            spline activation over the batch.
        """
        with torch.no_grad():
            bases = self.basis(x)  # [N, in, basis]
            # Per-edge activation: sum over basis, take absolute value
            edge_activation = torch.einsum(
                'oib,nib->noi',
                self.spline_weight * self.spline_scaler.unsqueeze(-1),
                bases
            )  # [N, out, in]
            importance = edge_activation.abs().mean(dim=0)  # [out, in]
        return importance

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'grid_size={self.grid_size}, '
            f'spline_order={self.spline_order}, '
            f'num_params={self.num_parameters}'
        )


class KANHead(nn.Module):
    """
    Multi-layer KAN head for classification or embedding transform.

    Supports:
    - Single layer: KAN [in_dim → out_dim]
    - Two layer: KAN [in_dim → hidden_dim] → KAN [hidden_dim → out_dim]

    Args:
        in_features:    Input embedding dimension (e.g., 512 from ResNet18)
        out_features:   Output dimension (e.g., num_classes or embedding dim)
        hidden_features: If not None, creates a 2-layer KAN with this hidden dim
        grid_size:      B-spline grid intervals
        spline_order:   B-spline order
        dropout:        Dropout rate between layers (for 2-layer)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Optional[int] = None,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        layers = []
        if hidden_features is not None:
            layers.append(KANLinear(in_features, hidden_features,
                                     grid_size=grid_size,
                                     spline_order=spline_order))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(KANLinear(hidden_features, out_features,
                                     grid_size=grid_size,
                                     spline_order=spline_order))
        else:
            layers.append(KANLinear(in_features, out_features,
                                     grid_size=grid_size,
                                     spline_order=spline_order))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_kan_layers(self):
        """Return list of KANLinear layers (for SFA attribution)."""
        return [m for m in self.modules() if isinstance(m, KANLinear)]
