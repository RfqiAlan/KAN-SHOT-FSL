"""
SFA: Spline-based Feature Attribution for KAN-ProtoNet

Provides interpretability for KAN-based few-shot classification by:
1. Computing per-dimension feature importance via KAN spline magnitudes
2. Evaluating attribution quality with Insertion/Deletion AUC metrics
3. Cross-checking with Grad-CAM for concordance analysis

Reference: Adapted from path-based KAN attribution (neuroimaging, OpenReview)
           and Petsiuk et al. (2018) Insertion/Deletion AUC framework.

Key difference from prior work: SFA operates in the few-shot metric-learning
setting, where attribution must account for prototype-relative decisions
rather than absolute class scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kan import KANLinear


@dataclass
class SFAResult:
    """Container for SFA attribution results."""
    feature_importance: torch.Tensor  # [feat_dim] — importance per dimension
    insertion_auc: float
    deletion_auc: float
    sufficiency_scores: List[float]  # sufficiency at each K
    top_k_dims: torch.Tensor  # indices of top-K important dimensions


class SplineFeatureAttribution:
    """
    Spline-based Feature Attribution (SFA) for KAN layers.

    Computes feature importance based on KAN spline magnitudes,
    then evaluates attribution quality via Insertion/Deletion metrics.

    Usage:
        sfa = SplineFeatureAttribution(kan_protonet_method)
        result = sfa.compute_attribution(z_q, centroids, y_q)
    """

    def __init__(self, method: nn.Module, num_steps: int = 10):
        """
        Args:
            method: KAN-ProtoNet method module containing KAN layers
            num_steps: Number of steps for insertion/deletion curves
        """
        self.method = method
        self.num_steps = num_steps

    def get_spline_importance(
        self,
        z: torch.Tensor,
        kan_layer: KANLinear
    ) -> torch.Tensor:
        """
        Compute per-input-dimension importance from KAN spline magnitudes.

        For a KAN layer with spline_weight [out, in, basis] and input x,
        importance_j = mean over batch and output dims of |Σ_b w_{o,j,b} * B_b(x_j)|

        Args:
            z: Input embeddings [N, in_features]
            kan_layer: KANLinear layer to analyze

        Returns:
            importance: [in_features] tensor of importance scores
        """
        return kan_layer.get_spline_importance(z).mean(dim=0)  # mean over output dims

    def compute_feature_ranking(
        self,
        z_q: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rank embedding dimensions by KAN spline importance.

        Args:
            z_q: Query embeddings [batch, q_shot, feat_dim] or [N, feat_dim]

        Returns:
            ranked_dims: Indices sorted by importance (most important first)
                         Shape: [feat_dim]
        """
        # Flatten batch dimensions
        if z_q.dim() == 3:
            z_flat = z_q.reshape(-1, z_q.size(-1))
        else:
            z_flat = z_q

        # Get importance from all KAN layers
        importance = torch.zeros(z_flat.size(-1), device=z_flat.device)

        # For transform mode
        if hasattr(self.method, 'kan_transform') and self.method.kan_transform is not None:
            for layer in self.method.kan_transform.get_kan_layers():
                imp = self.get_spline_importance(z_flat, layer)
                importance += imp

        # For distance mode — importance is on the difference vector,
        # which maps back to the same embedding dimensions
        if hasattr(self.method, 'kan_distance') and self.method.kan_distance is not None:
            # Use a zero-centered probe to get distance layer importance
            imp = self.get_spline_importance(z_flat, self.method.kan_distance)
            importance += imp

        # Sort by importance (descending)
        ranked_dims = importance.argsort(descending=True)
        return ranked_dims, importance

    def deletion_test(
        self,
        z_q: torch.Tensor,
        z_s: torch.Tensor,
        y_s: torch.Tensor,
        y_q: torch.Tensor,
        model: nn.Module,
        ranked_dims: torch.Tensor,
    ) -> Tuple[List[float], float]:
        """
        Deletion test: progressively zero-out top-K important dimensions.

        Good attribution → confidence drops quickly (low AUC).

        Args:
            z_q: Query embeddings [batch, q_shot, d]
            z_s: Support embeddings [batch, s_shot, d]
            y_s: Support labels [batch, s_shot]
            y_q: Query labels [batch, q_shot]
            model: Backbone model (not used here since we have embeddings)
            ranked_dims: Dimension indices sorted by importance

        Returns:
            confidences: List of mean correct-class confidence at each step
            auc: Area under the deletion curve (lower = better)
        """
        feat_dim = z_q.size(-1)
        step_size = max(1, feat_dim // self.num_steps)
        confidences = []

        for k in range(0, feat_dim + 1, step_size):
            z_q_masked = z_q.clone()
            z_s_masked = z_s.clone()

            if k > 0:
                dims_to_zero = ranked_dims[:k]
                z_q_masked[:, :, dims_to_zero] = 0
                z_s_masked[:, :, dims_to_zero] = 0

            # Run through KAN-ProtoNet with masked embeddings
            with torch.no_grad():
                log_probas = self._get_log_probas(z_s_masked, z_q_masked, y_s)

            # Get correct-class confidence
            probas = log_probas.exp()  # [batch, q, K]
            num_classes = probas.size(-1)
            correct_conf = self._get_correct_class_confidence(probas, y_q, num_classes)
            confidences.append(correct_conf.mean().item())

        # Compute AUC (trapezoidal rule)
        auc = np.trapz(confidences, dx=1.0 / len(confidences))
        return confidences, auc

    def insertion_test(
        self,
        z_q: torch.Tensor,
        z_s: torch.Tensor,
        y_s: torch.Tensor,
        y_q: torch.Tensor,
        model: nn.Module,
        ranked_dims: torch.Tensor,
    ) -> Tuple[List[float], float]:
        """
        Insertion test: start from zero, progressively restore top-K dims.

        Good attribution → confidence recovers quickly (high AUC).

        Args:
            Same as deletion_test.

        Returns:
            confidences: List of mean correct-class confidence at each step
            auc: Area under the insertion curve (higher = better)
        """
        feat_dim = z_q.size(-1)
        step_size = max(1, feat_dim // self.num_steps)
        confidences = []

        for k in range(0, feat_dim + 1, step_size):
            z_q_masked = torch.zeros_like(z_q)
            z_s_masked = torch.zeros_like(z_s)

            if k > 0:
                dims_to_restore = ranked_dims[:k]
                z_q_masked[:, :, dims_to_restore] = z_q[:, :, dims_to_restore]
                z_s_masked[:, :, dims_to_restore] = z_s[:, :, dims_to_restore]

            with torch.no_grad():
                log_probas = self._get_log_probas(z_s_masked, z_q_masked, y_s)

            probas = log_probas.exp()
            num_classes = probas.size(-1)
            correct_conf = self._get_correct_class_confidence(probas, y_q, num_classes)
            confidences.append(correct_conf.mean().item())

        auc = np.trapz(confidences, dx=1.0 / len(confidences))
        return confidences, auc

    def sufficiency_test(
        self,
        z_q: torch.Tensor,
        z_s: torch.Tensor,
        y_s: torch.Tensor,
        y_q: torch.Tensor,
        model: nn.Module,
        ranked_dims: torch.Tensor,
        k_values: Optional[List[int]] = None,
    ) -> Dict[int, float]:
        """
        Sufficiency test: keep only top-K dims, measure prediction accuracy.

        Args:
            k_values: List of K values to test. Default: [10%, 20%, ..., 100%]

        Returns:
            Dict mapping K → accuracy with only top-K dimensions
        """
        feat_dim = z_q.size(-1)
        if k_values is None:
            k_values = [int(feat_dim * p) for p in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]]

        results = {}
        for k in k_values:
            z_q_masked = torch.zeros_like(z_q)
            z_s_masked = torch.zeros_like(z_s)

            dims_to_keep = ranked_dims[:k]
            z_q_masked[:, :, dims_to_keep] = z_q[:, :, dims_to_keep]
            z_s_masked[:, :, dims_to_keep] = z_s[:, :, dims_to_keep]

            with torch.no_grad():
                log_probas = self._get_log_probas(z_s_masked, z_q_masked, y_s)

            preds = log_probas.exp().argmax(-1)
            acc = (preds == y_q).float().mean().item()
            results[k] = acc

        return results

    def full_evaluation(
        self,
        z_q: torch.Tensor,
        z_s: torch.Tensor,
        y_s: torch.Tensor,
        y_q: torch.Tensor,
        model: nn.Module,
    ) -> SFAResult:
        """
        Run complete SFA evaluation: ranking + deletion + insertion + sufficiency.

        Returns:
            SFAResult with all metrics
        """
        ranked_dims, importance = self.compute_feature_ranking(z_q)

        _, del_auc = self.deletion_test(z_q, z_s, y_s, y_q, model, ranked_dims)
        _, ins_auc = self.insertion_test(z_q, z_s, y_s, y_q, model, ranked_dims)
        sufficiency = self.sufficiency_test(z_q, z_s, y_s, y_q, model, ranked_dims)

        return SFAResult(
            feature_importance=importance,
            insertion_auc=ins_auc,
            deletion_auc=del_auc,
            sufficiency_scores=list(sufficiency.values()),
            top_k_dims=ranked_dims[:50],  # Top 50 dims
        )

    def _get_log_probas(
        self,
        z_s: torch.Tensor,
        z_q: torch.Tensor,
        y_s: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log-probabilities using the method's KAN metric.
        Dispatches to the correct formulation based on kan_mode.
        """
        from kan_shot.methods.utils import compute_centroids
        num_classes = y_s.unique().size(0)

        if self.method.kan_mode == 'distance':
            centroids = compute_centroids(z_s, y_s, n_way=num_classes)
            diff = z_q.unsqueeze(2) - centroids.unsqueeze(1)
            diff_flat = diff.view(-1, z_q.size(-1))
            scores_flat = self.method.kan_distance(diff_flat)
            scores = scores_flat.view(z_q.size(0), z_q.size(1), num_classes)
            logits = scores / self.method.temperature
            return logits.log_softmax(-1)
        else:
            raise NotImplementedError("SFA only fully supports distance mode right now")

    @staticmethod
    def _get_correct_class_confidence(
        probas: torch.Tensor,
        y_q: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """Extract confidence for the correct class."""
        # probas: [batch, q, K], y_q: [batch, q]
        batch_size = probas.size(0)
        q_shot = probas.size(1)
        # Gather correct class probabilities
        correct_conf = probas.gather(2, y_q.unsqueeze(-1)).squeeze(-1)  # [batch, q]
        return correct_conf


class RandomAttribution:
    """
    Random baseline for SFA comparison.
    Uses random feature ranking instead of spline-based.
    """

    def __init__(self, method: nn.Module, num_steps: int = 10):
        self.sfa = SplineFeatureAttribution(method, num_steps)

    def full_evaluation(
        self,
        z_q: torch.Tensor,
        z_s: torch.Tensor,
        y_s: torch.Tensor,
        y_q: torch.Tensor,
        model: nn.Module,
        num_trials: int = 5,
    ) -> SFAResult:
        """Run evaluation with random feature ranking (averaged over trials)."""
        feat_dim = z_q.size(-1)
        del_aucs = []
        ins_aucs = []

        for _ in range(num_trials):
            random_ranking = torch.randperm(feat_dim, device=z_q.device)
            _, del_auc = self.sfa.deletion_test(z_q, z_s, y_s, y_q, model, random_ranking)
            _, ins_auc = self.sfa.insertion_test(z_q, z_s, y_s, y_q, model, random_ranking)
            del_aucs.append(del_auc)
            ins_aucs.append(ins_auc)

        return SFAResult(
            feature_importance=torch.zeros(feat_dim),
            insertion_auc=np.mean(ins_aucs),
            deletion_auc=np.mean(del_aucs),
            sufficiency_scores=[],
            top_k_dims=torch.arange(feat_dim),
        )
