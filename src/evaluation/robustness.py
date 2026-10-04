"""
Robustness Evaluation for KAN-Fine++

Implements:
1. Histology-specific image corruptions (stain jitter, blur, noise, JPEG, brightness)
2. Feature-space adversarial attacks (FGSM, PGD)
3. Systematic evaluation across severity levels

Each corruption has 3 severity levels calibrated for histology images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import io
from PIL import Image
import torchvision.transforms.functional as TF


# ============================================================================
# Histology-Specific Image Corruptions
# ============================================================================

def gaussian_blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply Gaussian blur to batch of images.
    
    Args:
        images: [B, C, H, W] tensor
        sigma: Blur sigma
    Returns:
        Blurred images
    """
    # Kernel size must be odd
    kernel_size = int(4 * sigma + 0.5) * 2 + 1
    kernel_size = max(3, kernel_size)
    return TF.gaussian_blur(images, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])


def gaussian_noise(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add Gaussian noise to images.
    
    Args:
        images: [B, C, H, W] tensor (assumed normalized)
        sigma: Noise standard deviation
    """
    noise = torch.randn_like(images) * sigma
    return torch.clamp(images + noise, 0, 1)


def stain_jitter(images: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Simulate H&E stain variation by applying channel-wise color jitter.
    
    This approximates real stain variation between scanners/labs by
    independently scaling each color channel.
    
    Args:
        images: [B, C, H, W] tensor (RGB, normalized to [0,1])
        alpha: Maximum scale factor (higher = more variation)
    """
    B, C, H, W = images.shape
    # Random per-channel scale factor: 1 ± alpha
    scales = 1.0 + (torch.rand(B, C, 1, 1, device=images.device) * 2 - 1) * alpha
    return torch.clamp(images * scales, 0, 1)


def jpeg_compression(images: torch.Tensor, quality: int) -> torch.Tensor:
    """
    Simulate JPEG compression artifacts.
    
    Args:
        images: [B, C, H, W] tensor (normalized [0,1])
        quality: JPEG quality (1-100, lower = more compression)
    """
    device = images.device
    result = []
    for img in images:
        # Convert to PIL, compress, convert back
        img_pil = TF.to_pil_image(img.cpu().clamp(0, 1))
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img_compressed = Image.open(buffer)
        img_tensor = TF.to_tensor(img_compressed).to(device)
        result.append(img_tensor)
    return torch.stack(result)


def brightness_shift(images: torch.Tensor, delta: float) -> torch.Tensor:
    """
    Apply random brightness shift.
    
    Args:
        images: [B, C, H, W] tensor
        delta: Maximum brightness shift magnitude
    """
    B = images.shape[0]
    shifts = (torch.rand(B, 1, 1, 1, device=images.device) * 2 - 1) * delta
    return torch.clamp(images + shifts, 0, 1)


# Corruption registry with severity levels
CORRUPTION_REGISTRY = {
    'gaussian_blur': {
        'fn': gaussian_blur,
        'levels': {1: {'sigma': 0.5}, 2: {'sigma': 1.0}, 3: {'sigma': 2.0}},
    },
    'gaussian_noise': {
        'fn': gaussian_noise,
        'levels': {1: {'sigma': 0.01}, 2: {'sigma': 0.05}, 3: {'sigma': 0.10}},
    },
    'stain_jitter': {
        'fn': stain_jitter,
        'levels': {1: {'alpha': 0.05}, 2: {'alpha': 0.10}, 3: {'alpha': 0.20}},
    },
    'jpeg_compression': {
        'fn': jpeg_compression,
        'levels': {1: {'quality': 75}, 2: {'quality': 50}, 3: {'quality': 25}},
    },
    'brightness_shift': {
        'fn': brightness_shift,
        'levels': {1: {'delta': 0.1}, 2: {'delta': 0.2}, 3: {'delta': 0.3}},
    },
}


def apply_corruption(
    images: torch.Tensor,
    corruption_name: str,
    severity: int,
) -> torch.Tensor:
    """
    Apply a named corruption at a given severity level.
    
    Args:
        images: [B, C, H, W] tensor
        corruption_name: One of the keys in CORRUPTION_REGISTRY
        severity: 1, 2, or 3
        
    Returns:
        Corrupted images
    """
    if corruption_name not in CORRUPTION_REGISTRY:
        raise ValueError(f"Unknown corruption: {corruption_name}. "
                         f"Available: {list(CORRUPTION_REGISTRY.keys())}")
    if severity not in [1, 2, 3]:
        raise ValueError(f"Severity must be 1, 2, or 3, got {severity}")
    
    entry = CORRUPTION_REGISTRY[corruption_name]
    kwargs = entry['levels'][severity]
    return entry['fn'](images, **kwargs)


# ============================================================================
# Feature-Space Adversarial Attacks
# ============================================================================

def fgsm_attack(
    method: nn.Module,
    z_s: torch.Tensor,
    z_q: torch.Tensor,
    y_s: torch.Tensor,
    y_q: torch.Tensor,
    epsilon: float,
    model: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    FGSM attack on feature-space embeddings (attacks the head, not backbone).
    
    Computes adversarial perturbation on z_q to maximize classification loss.
    
    Args:
        method: Few-shot method (KAN-ProtoNet or baseline)
        z_s: Support embeddings [batch, s_shot, d]
        z_q: Query embeddings [batch, q_shot, d] — will be perturbed
        y_s: Support labels [batch, s_shot]
        y_q: Query labels [batch, q_shot]
        epsilon: Perturbation magnitude
        model: Not used (for API compatibility)
        
    Returns:
        Perturbed z_q: [batch, q_shot, d]
    """
    z_q_adv = z_q.clone().detach().requires_grad_(True)
    
    # Forward pass to get loss
    # We need to handle the case where method expects images but we have embeddings
    # For feature-space attack, we directly compute the metric
    num_classes = y_s.unique().size(0)
    
    if hasattr(method, 'kan_mode'):
        # KAN-ProtoNet: use internal forward functions
        if method.kan_mode == 'transform':
            log_probas = method._forward_transform(z_s, z_q_adv, y_s, num_classes)
        elif method.kan_mode == 'distance':
            log_probas = method._forward_distance(z_s, z_q_adv, y_s, num_classes)
        elif method.kan_mode == 'dual':
            log_probas = method._forward_dual(z_s, z_q_adv, y_s, num_classes)
    else:
        # Standard ProtoNet: L2 distance
        from src.methods.utils import compute_centroids, get_one_hot
        centroids = compute_centroids(z_s, y_s)
        l2_distance = (
            -2 * z_q_adv.matmul(centroids.transpose(1, 2))
            + (centroids ** 2).sum(2).unsqueeze(1)
            + (z_q_adv ** 2).sum(2).unsqueeze(-1)
        )
        log_probas = (-l2_distance).log_softmax(-1)
    
    # Compute loss (cross-entropy)
    from src.methods.utils import get_one_hot
    one_hot_q = get_one_hot(y_q, num_classes)
    loss = -(one_hot_q * log_probas).sum(-1).mean()
    
    # Backward to get gradients on z_q
    loss.backward()
    
    # FGSM perturbation
    grad_sign = z_q_adv.grad.sign()
    z_q_perturbed = z_q_adv.detach() + epsilon * grad_sign
    
    return z_q_perturbed


def pgd_attack(
    method: nn.Module,
    z_s: torch.Tensor,
    z_q: torch.Tensor,
    y_s: torch.Tensor,
    y_q: torch.Tensor,
    epsilon: float,
    num_steps: int = 10,
    step_size: Optional[float] = None,
    model: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    PGD attack on feature-space embeddings.
    
    Iterative version of FGSM with projection onto ε-ball.
    
    Args:
        method: Few-shot method
        z_s, z_q, y_s, y_q: Episode data
        epsilon: Max perturbation magnitude
        num_steps: Number of PGD iterations
        step_size: Per-step size (default: epsilon / num_steps * 2)
        model: Not used (for API compatibility)
        
    Returns:
        Perturbed z_q
    """
    if step_size is None:
        step_size = epsilon / num_steps * 2.5
    
    z_q_adv = z_q.clone().detach()
    z_q_orig = z_q.clone().detach()
    
    for _ in range(num_steps):
        z_q_adv.requires_grad_(True)
        
        num_classes = y_s.unique().size(0)
        
        if hasattr(method, 'kan_mode'):
            if method.kan_mode == 'transform':
                log_probas = method._forward_transform(z_s, z_q_adv, y_s, num_classes)
            elif method.kan_mode == 'distance':
                log_probas = method._forward_distance(z_s, z_q_adv, y_s, num_classes)
            elif method.kan_mode == 'dual':
                log_probas = method._forward_dual(z_s, z_q_adv, y_s, num_classes)
        else:
            from src.methods.utils import compute_centroids, get_one_hot
            centroids = compute_centroids(z_s, y_s)
            l2_distance = (
                -2 * z_q_adv.matmul(centroids.transpose(1, 2))
                + (centroids ** 2).sum(2).unsqueeze(1)
                + (z_q_adv ** 2).sum(2).unsqueeze(-1)
            )
            log_probas = (-l2_distance).log_softmax(-1)
        
        from src.methods.utils import get_one_hot
        one_hot_q = get_one_hot(y_q, num_classes)
        loss = -(one_hot_q * log_probas).sum(-1).mean()
        loss.backward()
        
        # PGD step
        grad_sign = z_q_adv.grad.sign()
        z_q_adv = z_q_adv.detach() + step_size * grad_sign
        
        # Project back onto epsilon-ball
        perturbation = z_q_adv - z_q_orig
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        z_q_adv = z_q_orig + perturbation
    
    return z_q_adv.detach()


# ============================================================================
# Evaluation Harness
# ============================================================================

@dataclass
class RobustnessResult:
    """Result of robustness evaluation for one model."""
    model_name: str
    clean_accuracy: float
    corruption_results: Dict[str, Dict[int, float]]  # corruption → severity → accuracy
    adversarial_results: Dict[str, Dict[float, float]]  # attack → epsilon → accuracy
    
    def average_corruption_drop(self) -> float:
        """Average accuracy drop across all corruptions and severities."""
        drops = []
        for corruption, levels in self.corruption_results.items():
            for severity, acc in levels.items():
                drops.append(self.clean_accuracy - acc)
        return np.mean(drops) if drops else 0.0


def evaluate_corruption_robustness(
    method: nn.Module,
    model: nn.Module,
    episode_loader,
    corruptions: Optional[List[str]] = None,
    severities: List[int] = [1, 2, 3],
    num_episodes: int = 200,
    device: str = 'cuda',
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate model under various image corruptions.
    
    Applies corruptions to query images before feature extraction,
    then runs few-shot classification.
    
    Args:
        method: Few-shot method
        model: Backbone model
        episode_loader: DataLoader yielding (support, query, s_labels, q_labels)
        corruptions: List of corruption names (default: all)
        severities: List of severity levels
        num_episodes: Number of evaluation episodes
        device: Device
        
    Returns:
        Dict mapping corruption_name → {severity → accuracy}
    """
    if corruptions is None:
        corruptions = list(CORRUPTION_REGISTRY.keys())
    
    results = {}
    model.eval()
    method.eval()
    
    for corruption in corruptions:
        results[corruption] = {}
        for severity in severities:
            correct = 0
            total = 0
            
            for i, data in enumerate(episode_loader):
                if i >= num_episodes:
                    break
                    
                support, query, s_labels, q_labels = data
                support = support.to(device)
                s_labels = s_labels.to(device)
                q_labels = q_labels.to(device)
                
                # Apply corruption to query images
                # query shape: [batch, q_shot, C, H, W]
                batch, q_shot = query.shape[:2]
                query_flat = query.reshape(-1, *query.shape[2:]).to(device)
                query_corrupted = apply_corruption(query_flat, corruption, severity)
                query_corrupted = query_corrupted.reshape(batch, q_shot, *query.shape[2:])
                
                with torch.no_grad():
                    _, preds = method(support, query_corrupted, s_labels, q_labels, model)
                
                correct += (preds == q_labels).float().sum().item()
                total += q_labels.numel()
            
            results[corruption][severity] = correct / total if total > 0 else 0.0
    
    return results
