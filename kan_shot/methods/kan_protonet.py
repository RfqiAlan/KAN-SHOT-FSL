import torch
import torch.nn as nn
import argparse
from typing import Tuple, Optional

from kan_shot.kan import KANLinear, KANHead
from kan_shot.methods.utils import (
    extract_features,
    compute_metric_logits,
    compute_prototypical_loss,
    get_one_hot,
    compute_centroids
)
from kan_shot.methods.method import FSmethod


class KANProtoNet(FSmethod):
    """
    KAN-ProtoNet: Prototypical Network with KAN-based metric head.
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # General config
        self.distance_metric = getattr(args, 'distance_metric', 'cosine')
        self.temperature = getattr(args, 'temperature', 1.0)
        self.label_smoothing = getattr(args, 'label_smoothing', 0.1)
        self.n_way = getattr(args, 'n_way', 9)

        # KAN config
        self.kan_mode = getattr(args, 'kan_mode', 'transform')  # 'transform', 'distance', 'dual'
        self.feat_dim = getattr(args, 'feat_dim', 512)
        self.kan_hidden = getattr(args, 'kan_hidden', None)
        self.kan_out_dim = getattr(args, 'kan_out_dim', 512)
        self.grid_size = getattr(args, 'kan_grid_size', 5)
        self.spline_order = getattr(args, 'kan_spline_order', 3)
        self.kan_dropout = getattr(args, 'kan_dropout', 0.0)
        self.kan_prenorm = getattr(args, 'kan_prenorm', False)

        # Build KAN components
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
        
        assert set(y_s.unique().tolist()) == set(range(self.n_way)), "Support labels are not properly remapped to [0, n_way-1]"
        assert set(y_q.unique().tolist()).issubset(set(range(self.n_way))), "Query labels contain out-of-bounds classes"

        with torch.set_grad_enabled(self.training and model.training):
            z_s = extract_features(x_s, model)
            z_q = extract_features(x_q, model)

        if self.kan_prenorm:
            import torch.nn.functional as F
            z_s = F.normalize(z_s, p=2, dim=-1)
            z_q = F.normalize(z_q, p=2, dim=-1)

        if self.kan_mode == 'transform':
            z_s_kan = self.kan_transform(z_s)
            z_q_kan = self.kan_transform(z_q)

            logits = compute_metric_logits(
                z_support=z_s_kan,
                z_query=z_q_kan,
                y_support=y_s,
                n_way=self.n_way,
                distance_metric=self.distance_metric,
                temperature=self.temperature
            )

            # Debug printing for first step
            if self.training and getattr(self, '_first_step_dist', True):
                print(f"\n🔍 [KAN Debug] Logits range: min={logits.min().item():.4f}, max={logits.max().item():.4f}")
                self._first_step_dist = False

        elif self.kan_mode == 'distance':
            # This bypasses compute_metric_logits because KAN distance is fundamentally different.
            centroids = compute_centroids(z_s, y_s, n_way=self.n_way)
            diff = z_q.unsqueeze(2) - centroids.unsqueeze(1)
            diff_flat = diff.view(-1, self.feat_dim)
            scores_flat = self.kan_distance(diff_flat)
            scores = scores_flat.view(z_q.size(0), z_q.size(1), self.n_way)
            logits = scores / self.temperature

        elif self.kan_mode == 'dual':
            z_s_kan = self.kan_transform(z_s)
            z_q_kan = self.kan_transform(z_q)
            centroids = compute_centroids(z_s_kan, y_s, n_way=self.n_way)
            diff = z_q_kan.unsqueeze(2) - centroids.unsqueeze(1)
            diff_flat = diff.view(-1, self.kan_out_dim)
            scores_flat = self.kan_distance(diff_flat)
            scores = scores_flat.view(z_q_kan.size(0), z_q_kan.size(1), self.n_way)
            logits = scores / self.temperature

        loss = compute_prototypical_loss(
            logits=logits,
            targets=y_q,
            label_smoothing=self.label_smoothing
        )

        preds_q = logits.detach().argmax(dim=-1)

        return loss, preds_q
