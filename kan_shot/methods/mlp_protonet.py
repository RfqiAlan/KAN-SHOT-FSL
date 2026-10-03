import torch
import torch.nn as nn
import argparse
from typing import Tuple, Optional

from kan_shot.methods.utils import (
    extract_features,
    compute_metric_logits,
    compute_prototypical_loss
)
from kan_shot.methods.method import FSmethod


class MLPProtoNet(FSmethod):
    """
    MLP-ProtoNet: Prototypical Network with MLP/Linear projection layer.
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        
        self.distance_metric = getattr(args, 'distance_metric', 'cosine')
        self.temperature = getattr(args, 'temperature', 1.0)
        self.label_smoothing = getattr(args, 'label_smoothing', 0.1)
        self.n_way = getattr(args, 'n_way', 9)

        self.feat_dim = getattr(args, 'feat_dim', 512)
        self.out_dim = getattr(args, 'kan_out_dim', 512)
        self.hidden_dim = getattr(args, 'kan_hidden', None)

        if self.hidden_dim is not None:
            self.transform = nn.Sequential(
                nn.Linear(self.feat_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.out_dim)
            )
        else:
            self.transform = nn.Linear(self.feat_dim, self.out_dim)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.transform.parameters())

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

        # Note: model should be frozen, but we enable grad if method is training
        with torch.set_grad_enabled(self.training and model.training):
            z_s = extract_features(x_s, model)
            z_q = extract_features(x_q, model)

        z_s = self.transform(z_s)
        z_q = self.transform(z_q)

        logits = compute_metric_logits(
            z_support=z_s,
            z_query=z_q,
            y_support=y_s,
            n_way=self.n_way,
            distance_metric=self.distance_metric,
            temperature=self.temperature
        )

        loss = compute_prototypical_loss(
            logits=logits,
            targets=y_q,
            label_smoothing=self.label_smoothing
        )

        preds_q = logits.detach().argmax(dim=-1)

        return loss, preds_q
