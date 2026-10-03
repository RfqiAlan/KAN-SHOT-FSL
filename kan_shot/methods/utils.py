import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

def get_one_hot(y_s: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    args:
        y_s : torch.Tensor of shape [n_task, shot]
    returns
        y_s : torch.Tensor of shape [n_task, shot, num_classes]
    """
    one_hot_size = list(y_s.size()) + [num_classes]
    one_hot = torch.zeros(one_hot_size, device=y_s.device)
    one_hot.scatter_(-1, y_s.unsqueeze(-1), 1)
    return one_hot

def compute_centroids(z_s: torch.Tensor, y_s: torch.Tensor, n_way: int) -> torch.Tensor:
    """
    inputs:
        z_s : torch.Tensor of size [batch_size, s_shot, d]
        y_s : torch.Tensor of size [batch_size, s_shot]
        n_way: number of classes

    updates :
        centroids : torch.Tensor of size [batch_size, n_way, d]
    """
    one_hot = get_one_hot(y_s, num_classes=n_way).transpose(1, 2)  # [batch, n_way, s_shot]
    centroids = one_hot.bmm(z_s) / (one_hot.sum(-1, keepdim=True) + 1e-8)  # [batch, n_way, d]
    return centroids

def extract_features(x: Tensor, model: nn.Module) -> torch.Tensor:
    """
    Extract features from support and query set using the provided model.
    """
    batch, shot = x.size()[:2]
    feat_dim = x.size()[-3:]
    z = model(x.view(batch * shot, *feat_dim))
    z = z.view(batch, shot, -1)  # [batch, shot, d]
    return z

def compute_metric_logits(
    z_support: torch.Tensor,
    z_query: torch.Tensor,
    y_support: torch.Tensor,
    n_way: int,
    distance_metric: str = "cosine",
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Unified metric computation for few-shot evaluation.
    """
    if distance_metric == "cosine":
        # Normalize embeddings
        z_support = F.normalize(z_support, p=2, dim=-1)
        z_query = F.normalize(z_query, p=2, dim=-1)

        # Compute prototypes
        prototypes = compute_centroids(z_support, y_support, n_way=n_way)
        
        # Normalize prototypes
        prototypes = F.normalize(prototypes, p=2, dim=-1)

        # Cosine similarity
        logits = torch.matmul(z_query, prototypes.transpose(-1, -2))

    elif distance_metric == "l2":
        prototypes = compute_centroids(z_support, y_support, n_way=n_way)

        # Negative L2 squared distance
        logits = -(
            (z_query.unsqueeze(-2) - prototypes.unsqueeze(-3))
            .pow(2)
            .sum(dim=-1)
        )

    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    return logits / temperature

def compute_prototypical_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing: float = 0.0
) -> torch.Tensor:
    """
    Computes cross entropy loss with optional label smoothing.
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        label_smoothing=label_smoothing
    )
