"""
SFA-Guided Pathology Attention Map Visualization
=================================================
Synthesizes spatial attribution maps by projecting KAN Spline Feature Attribution (SFA)
weights back onto the ResNet18 spatial feature maps (layer4), generating a 3-Panel Triptych:
  - Panel A: Original Histopathology Patch (H&E, 224x224)
  - Panel B: SFA-Guided Attention Map Overlay (Magma/Jet heatmap at 50% alpha)
  - Panel C: 1D Learned KAN Spline Curve phi_j(Delta) for the top SFA dimension

Usage:
  python scripts/visualize_sfa_spatial.py \
      --backbone checkpoints/seed2021/resnet18_nct.pth \
      --kan_metric checkpoints/seed2021/kanprotonet_cosine_nct.pth \
      --manifest episodes/crc_val_shot5_seed2021.json \
      --colormap magma \
      --alpha 0.5 \
      --output_dir results/sfa_pathology
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from kan_shot.methods import KANProtoNet
from kan_shot.methods.utils import compute_centroids


# ---------------------------------------------------------------------------
# Image Preprocessing & De-normalization
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_eval_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized PyTorch tensor [3, H, W] back to RGB float [H, W, 3] in [0, 1]."""
    t = tensor.clone()
    for ch, m, s in zip(t, IMAGENET_MEAN, IMAGENET_STD):
        ch.mul_(s).add_(m)
    return t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


# ---------------------------------------------------------------------------
# Model Builders
# ---------------------------------------------------------------------------
def load_backbone(backbone_path: str, device: torch.device):
    """Load ResNet-18 backbone with weights, removing final fc."""
    model = models.resnet18(weights=None)
    model.fc = nn.Identity()
    raw = torch.load(backbone_path, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw)
    state = {k: v for k, v in state.items()
             if not k.startswith("fc.") and not k.endswith(("fc.weight", "fc.bias",
                                                            "classifier.weight", "classifier.bias"))}
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    # Extract conv body (up to layer4, excluding avgpool and fc)
    conv_body = nn.Sequential(*list(model.children())[:-2]).to(device)
    conv_body.eval()
    return model, conv_body


def load_kan_method(metric_path: str, device: torch.device):
    """Load trained KANProtoNet metric head."""
    ckpt = torch.load(metric_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    m_args = argparse.Namespace(**config)
    method = KANProtoNet(m_args)
    method.load_state_dict(ckpt["method_state_dict"], strict=True)
    method = method.to(device)
    method.eval()
    return method, m_args


# ---------------------------------------------------------------------------
# SFA Computation & Spline Evaluation
# ---------------------------------------------------------------------------
def compute_sfa_scores(kan_distance_layer, z_query: torch.Tensor = None, centroid: torch.Tensor = None, mode: str = "global"):
    """
    Computes SFA importance vector A in [0, 1]^D.
    
    If mode == 'global':
        Importance = |base_weight| + |spline_scaler| * mean(|spline_weight|)
    If mode == 'instance' and z_query, centroid provided:
        Computes the activation magnitude for Delta = z_query - centroid
    """
    kan_linear = kan_distance_layer
    spline_w = kan_linear.spline_weight.detach()[0]  # [512, num_basis]
    spline_scaler = kan_linear.spline_scaler.detach()[0]  # [512]
    base_w = kan_linear.base_weight.detach()[0] if kan_linear.base_weight is not None else torch.zeros(512, device=spline_w.device)

    if mode == "instance" and z_query is not None and centroid is not None:
        delta = (z_query - centroid).view(1, -1)  # [1, 512]
        bases = kan_linear.basis(delta)  # [1, 512, num_basis]
        # Spline activation per dim
        spline_act = (bases * spline_w.unsqueeze(0) * spline_scaler.unsqueeze(0).unsqueeze(-1)).sum(dim=-1).squeeze(0)
        # Base activation
        silu_act = F.silu(delta).squeeze(0) * base_w
        raw_importance = torch.abs(spline_act) + torch.abs(silu_act)
    else:
        spline_l1 = torch.mean(torch.abs(spline_w), dim=1)
        raw_importance = torch.abs(base_w) + torch.abs(spline_scaler) * spline_l1

    # Normalize to [0, 1] range for color mapping & weighting
    min_val = raw_importance.min()
    max_val = raw_importance.max()
    A_norm = (raw_importance - min_val) / (max_val - min_val + 1e-8)
    return A_norm, raw_importance


def evaluate_spline_curve(kan_distance_layer, dim_idx: int, grid_range=(-2.0, 2.0), num_points=200):
    """
    Evaluates 1D distance response phi_j(Delta) for a specific dimension.
    
    Returns:
        x_pts (numpy [num_points]): feature difference Delta
        total_eval (numpy [num_points]): phi_j(Delta)
        base_eval (numpy [num_points]): residual linear/SiLU component
        spline_eval (numpy [num_points]): B-spline component
    """
    kan_linear = kan_distance_layer
    device = kan_linear.spline_weight.device
    grid_points = torch.linspace(grid_range[0], grid_range[1], num_points, device=device).unsqueeze(1)  # [N, 1]
    
    basis_vals = kan_linear.basis(grid_points)  # [N, 1, num_basis]
    w_dim = kan_linear.spline_weight.detach()[0, dim_idx].unsqueeze(0).unsqueeze(0)  # [1, 1, num_basis]
    spline_out = (basis_vals * w_dim).sum(dim=-1).squeeze().cpu().numpy()  # [N]
    
    scaler = kan_linear.spline_scaler.detach()[0, dim_idx].item()
    base_w = kan_linear.base_weight.detach()[0, dim_idx].item() if kan_linear.base_weight is not None else 0.0

    x_np = grid_points.squeeze().cpu().numpy()
    silu_vals = x_np * (1.0 / (1.0 + np.exp(-np.clip(x_np, -15, 15))))
    base_eval = base_w * silu_vals
    total_eval = (scaler * spline_out) + base_eval
    spline_scaled = scaler * spline_out

    return x_np, total_eval, base_eval, spline_scaled


# ---------------------------------------------------------------------------
# Heatmap Synthesis (Tahap 2)
# ---------------------------------------------------------------------------
def synthesize_sfa_heatmap(Z_feat: torch.Tensor, A_scores: torch.Tensor, target_size=(224, 224)):
    """
    Spatial Weighted Aggregation:
        M = sum_j (A_j * Z_j)
    Followed by bilinear upsampling and min-max normalization.
    
    Args:
        Z_feat: Feature map tensor [1, D, H, W] from conv_body
        A_scores: Normalized SFA scores [D] in [0, 1]
        target_size: Target resolution (H_orig, W_orig)
        
    Returns:
        M_norm: [target_size[0], target_size[1]] numpy array in [0, 1]
    """
    # Z_feat: [1, 512, 7, 7], A_scores: [512]
    A_view = A_scores.view(1, -1, 1, 1).to(Z_feat.device)
    
    # Positive spatial aggregation (features post-ReLU are non-negative)
    M = (A_view * Z_feat).sum(dim=1, keepdim=True)  # [1, 1, H, W]
    
    # Bilinear interpolation to original image size
    M_up = F.interpolate(M, size=target_size, mode='bilinear', align_corners=False)
    M_np = M_up.squeeze().detach().cpu().numpy()
    
    # Min-max normalization for precise visual contrast
    min_val, max_val = M_np.min(), M_np.max()
    M_norm = (M_np - min_val) / (max_val - min_val + 1e-8)
    return M_norm


def overlay_heatmap(rgb_img: np.ndarray, heatmap_norm: np.ndarray, colormap_name: str = "magma", alpha: float = 0.5):
    """
    Overlays a normalized 2D heatmap [0, 1] onto an RGB image [0, 1].
    """
    cmap = plt.get_cmap(colormap_name)
    heatmap_colored = cmap(heatmap_norm)[:, :, :3]  # RGB [H, W, 3]
    blended = (1.0 - alpha) * rgb_img + alpha * heatmap_colored
    blended = np.clip(blended, 0.0, 1.0)
    return blended, heatmap_colored


# ---------------------------------------------------------------------------
# Pathology Clinical Descriptions
# ---------------------------------------------------------------------------
PATHOLOGY_META = {
    "TUM": {
        "title": "Colorectal Adenocarcinoma (Epithelium)",
        "morphology": "Atypical glandular crypt architecture, hyperchromatic nuclei, cribriform growth pattern.",
        "clinical_focus": "Structural crypt deformation & epithelial lumen irregularity driving distance metric."
    },
    "NORM": {
        "title": "Normal Colon Mucosa",
        "morphology": "Organized parallel crypts, intact goblet cell vacuoles, uniform nuclear polarity.",
        "clinical_focus": "Symmetric glandular spacing and basal crypt alignment."
    },
    "STR": {
        "title": "Cancer-Associated Stroma",
        "morphology": "Desmoplastic collagen fibers, spindle-shaped activated fibroblasts.",
        "clinical_focus": "Fibroblastic density and extracellular matrix reorganization."
    },
    "LYM": {
        "title": "Lymphocyte Infiltrate",
        "morphology": "Dense clusters of small round immune cells with dark spherical nuclei.",
        "clinical_focus": "Stromal tumor-infiltrating lymphocytes (TILs) concentration."
    },
    "MUC": {
        "title": "Colloid Mucus Pool",
        "morphology": "Acellular or sparsely cellular mucinous pools with floating tumor clusters.",
        "clinical_focus": "Extracellular mucin lake morphology."
    },
    "MUS": {
        "title": "Smooth Muscle Wall",
        "morphology": "Elongated smooth muscle bundles with eosinophilic cytoplasm.",
        "clinical_focus": "Muscularis propria orientation."
    },
    "ADI": {
        "title": "Adipose Tissue",
        "morphology": "Thin-walled signet cells with empty lipid droplets.",
        "clinical_focus": "Lipid vacuole architecture."
    },
    "DEB": {
        "title": "Necrotic Debris",
        "morphology": "Amorphous eosinophilic material, karyorrhectic fragments.",
        "clinical_focus": "Tumor necrosis and cellular breakdown."
    },
    "BACK": {
        "title": "Glass Background",
        "morphology": "Acellular clear slide region.",
        "clinical_focus": "Non-tissue background."
    }
}


# ---------------------------------------------------------------------------
# 3-Panel Triptych Plotting (Tahap 3 & 4)
# ---------------------------------------------------------------------------
def generate_triptych_figure(
    rgb_img: np.ndarray,
    heatmap_norm: np.ndarray,
    kan_distance_layer,
    top_dim_idx: int,
    sfa_rank: int,
    sfa_score: float,
    class_code: str,
    output_path: str,
    colormap_name: str = "magma",
    alpha: float = 0.5,
    query_delta: float = None,
    draw_contour: bool = False,
    episode_id: int = 0
):
    """
    Generates a publication-grade 3-Panel Triptych:
      - Panel A: Original H&E Patch
      - Panel B: SFA-Guided Attention Overlay
      - Panel C: 1D Learned KAN Spline Curve
    """
    blended_img, colored_map = overlay_heatmap(rgb_img, heatmap_norm, colormap_name, alpha)
    x_np, total_eval, base_eval, spline_scaled = evaluate_spline_curve(kan_distance_layer, top_dim_idx)

    # Style configuration
    plt.rcParams["font.family"] = "sans-serif"
    fig = plt.figure(figsize=(18, 5.5), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.28)

    meta = PATHOLOGY_META.get(class_code, {
        "title": class_code,
        "morphology": "Histopathological tissue features.",
        "clinical_focus": "Feature attribution driven classification."
    })

    # -------------------------------------------------------------------------
    # Panel A: Original Histopathology Patch
    # -------------------------------------------------------------------------
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(rgb_img)
    ax0.set_title(f"Panel A: Original H&E Patch\n[{class_code}] {meta['title']}", fontsize=12, fontweight="bold", pad=10)
    ax0.axis("off")
    for spine in ax0.spines.values():
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(1.0)

    # Scale bar annotation / metadata box
    ax0.text(0.03, 0.04, "Resolution: 224×224 (0.5µm/px)", transform=ax0.transAxes,
             fontsize=8.5, color="white", weight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6, edgecolor="none"))

    # -------------------------------------------------------------------------
    # Panel B: SFA-Guided Attention Map Overlay
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(blended_img)
    if draw_contour:
        cs = ax1.contour(heatmap_norm, levels=4, colors="white", linewidths=0.7, alpha=0.7)
        ax1.clabel(cs, inline=True, fontsize=6.5, fmt="%.2f")

    ax1.set_title(f"Panel B: SFA-Guided Attention Map\nWeighted Aggregation (α={alpha}, {colormap_name.title()})",
                  fontsize=12, fontweight="bold", pad=10)
    ax1.axis("off")

    # Inset Colorbar
    cax = fig.add_axes([ax1.get_position().x0 + 0.02, ax1.get_position().y0 + 0.03,
                        ax1.get_position().width * 0.4, 0.025])
    norm = Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap_name), norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=7, colors="white")
    cbar.set_label("SFA Intensity", fontsize=7.5, color="white", labelpad=2)
    cax.set_facecolor("black")
    cax.patch.set_alpha(0.5)

    # Annotation badge for top SFA driving score
    ax1.text(0.03, 0.94, f"Top Dim {top_dim_idx} (SFA={sfa_score:.2f})", transform=ax1.transAxes,
             fontsize=8.5, color="white", weight="bold", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#990000", alpha=0.75, edgecolor="none"))

    # -------------------------------------------------------------------------
    # Panel C: 1D Learned KAN Spline Curve
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Curves
    ax2.plot(x_np, total_eval, color="#8b0000", linewidth=2.8, label=r"Total KAN Response $\phi_j(\Delta)$", zorder=4)
    ax2.plot(x_np, spline_scaled, color="#20639b", linewidth=1.6, linestyle="-.", label="Spline Component (B-spline)", zorder=3)
    ax2.plot(x_np, base_eval, color="#666666", linewidth=1.2, linestyle="--", label="Base Linear/SiLU Path", zorder=2)

    # Reference axes
    ax2.axvline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
    ax2.axhline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.4)

    # Highlight current sample's query-prototype difference if provided
    if query_delta is not None and not np.isnan(query_delta):
        q_clamped = np.clip(query_delta, x_np.min(), x_np.max())
        idx_pt = np.argmin(np.abs(x_np - q_clamped))
        q_score = total_eval[idx_pt]

        ax2.scatter([q_clamped], [q_score], color="#ffd700", edgecolor="#b8860b", s=120, zorder=6,
                    label=f"Query Patch (Δ={query_delta:+.2f})")
        ax2.annotate(f"Query operating point\nΔ={query_delta:+.2f}, score={q_score:+.2f}",
                     xy=(q_clamped, q_score),
                     xytext=(q_clamped + (0.3 if q_clamped < 0.5 else -0.9), q_score - 0.25),
                     arrowprops=dict(facecolor="black", shrink=0.08, width=1.0, headwidth=5),
                     fontsize=8.0, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff9db", edgecolor="#e67700", alpha=0.9))

    ax2.set_title(f"Panel C: Learned Spline Metric $\\phi_{{{top_dim_idx}}}$\nResNet18 Dim #{top_dim_idx} (Rank #{sfa_rank}, SFA Score={sfa_score:.3f})",
                  fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel(r"Feature Difference $\Delta_j = (z_{query} - c_{proto})_j$", fontsize=10, fontweight="bold")
    ax2.set_ylabel(r"Metric Logit Contribution $\phi_j(\Delta_j)$", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.legend(loc="lower left", fontsize=8.0, framealpha=0.9)
    ax2.set_xlim([x_np.min(), x_np.max()])

    # -------------------------------------------------------------------------
    # Caption and Clinical Context (Bottom Subtitle)
    # -------------------------------------------------------------------------
    caption_text = (
        f"Figure: SFA-Guided Pathology Interpretability ({class_code} — {meta['title']}). "
        f"The attribution heatmap (Panel B) aggregates ResNet18 feature maps weighted by KAN spline importance scores (SFA={sfa_score:.2f}), "
        f"identifying {meta['clinical_focus'].lower()} "
        f"Panel C shows the exact 1D spline function $\\phi_{{{top_dim_idx}}}$ learned by KANProtoNet, demonstrating a non-linear penalty for deviations "
        f"from prototypical tissue morphology."
    )

    fig.text(0.5, -0.05, caption_text, ha="center", va="top", fontsize=9.2, style="italic", wrap=True)

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Generated Triptych Figure: {output_path}")

    # Also save metadata/caption sidecar
    caption_path = Path(output_path).with_suffix(".txt")
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(caption_text + "\n\n")
        f.write(f"Class: {class_code} ({meta['title']})\n")
        f.write(f"Morphological Features: {meta['morphology']}\n")
        f.write(f"Clinical Attribution: {meta['clinical_focus']}\n")
        f.write(f"Top Dimension: {top_dim_idx} (Rank {sfa_rank}, SFA Normalized: {sfa_score:.4f})\n")


# ---------------------------------------------------------------------------
# Multi-Class Comparative Grid Figure
# ---------------------------------------------------------------------------
def generate_multiclass_grid(
    samples_data: list,
    output_path: str,
    colormap_name: str = "magma",
    alpha: float = 0.5
):
    """
    Generates a publication summary grid comparing multiple tissue classes side-by-side
    (e.g., TUM vs NORM vs STR).
    """
    n_samples = len(samples_data)
    fig, axes = plt.subplots(n_samples, 3, figsize=(14, 4.2 * n_samples), dpi=300)
    if n_samples == 1:
        axes = axes.reshape(1, 3)

    plt.rcParams["font.family"] = "sans-serif"

    for row, data in enumerate(samples_data):
        rgb_img = data["rgb_img"]
        heatmap_norm = data["heatmap_norm"]
        class_code = data["class_code"]
        top_dim = data["top_dim"]
        sfa_score = data["sfa_score"]
        kan_distance_layer = data["kan_layer"]
        query_delta = data.get("query_delta", None)

        meta = PATHOLOGY_META.get(class_code, {"title": class_code})
        blended, _ = overlay_heatmap(rgb_img, heatmap_norm, colormap_name, alpha)

        # Col 0: Original
        axes[row, 0].imshow(rgb_img)
        axes[row, 0].set_title(f"[{class_code}] {meta['title']}\nOriginal H&E Patch", fontsize=11, fontweight="bold")
        axes[row, 0].axis("off")

        # Col 1: Overlay
        axes[row, 1].imshow(blended)
        axes[row, 1].set_title(f"SFA Attention Map (Top Dim {top_dim})\nScore={sfa_score:.2f} ({colormap_name.title()})",
                               fontsize=11, fontweight="bold")
        axes[row, 1].axis("off")

        # Col 2: Spline
        ax_sp = axes[row, 2]
        x_np, total_eval, base_eval, spline_scaled = evaluate_spline_curve(kan_distance_layer, top_dim)
        ax_sp.plot(x_np, total_eval, color="#8b0000", linewidth=2.4, label=r"$\phi(\Delta)$ Total")
        ax_sp.plot(x_np, spline_scaled, color="#20639b", linewidth=1.4, linestyle="-.", label="Spline")
        ax_sp.plot(x_np, base_eval, color="#666666", linewidth=1.0, linestyle="--", label="Base SiLU")
        ax_sp.axvline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

        if query_delta is not None and not np.isnan(query_delta):
            q_clamped = np.clip(query_delta, x_np.min(), x_np.max())
            idx_pt = np.argmin(np.abs(x_np - q_clamped))
            ax_sp.scatter([q_clamped], [total_eval[idx_pt]], color="#ffd700", edgecolor="#b8860b", s=70, zorder=5)

        ax_sp.set_title(f"Spline Metric $\\phi_{{{top_dim}}}$ (Dim #{top_dim})", fontsize=11, fontweight="bold")
        ax_sp.set_xlabel(r"$\Delta_j = (z_{query} - c_{proto})_j$", fontsize=9)
        ax_sp.set_ylabel(r"Score Contribution", fontsize=9)
        ax_sp.grid(True, linestyle="--", alpha=0.4)
        if row == 0:
            ax_sp.legend(loc="lower left", fontsize=7.5)

    plt.suptitle("SFA-Guided Pathology Attention Maps across Colorectal Tissue Classes",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Generated Multi-Class Summary Grid: {output_path}")


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SFA-Guided Pathology Attention Map Visualization")
    parser.add_argument("--backbone", type=str, default="checkpoints/seed2021/resnet18_nct.pth",
                        help="Path to trained ResNet18 backbone checkpoint")
    parser.add_argument("--kan_metric", type=str, default="checkpoints/seed2021/kanprotonet_cosine_nct.pth",
                        help="Path to trained KANProtoNet checkpoint")
    parser.add_argument("--manifest", type=str, default="episodes/crc_val_shot5_seed2021.json",
                        help="Path to episode manifest JSON")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Optional direct image path for standalone visualization")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Target image size (default 224 for histopathology)")
    parser.add_argument("--colormap", type=str, default="magma", choices=["magma", "jet", "inferno", "plasma", "viridis"],
                        help="Colormap for heatmap overlay (magma or jet recommended)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Heatmap overlay alpha transparency (default: 0.5)")
    parser.add_argument("--contour", action="store_true",
                        help="Draw subtle white contour lines on top of heatmap")
    parser.add_argument("--top_dim", type=int, default=None,
                        help="Explicit feature dimension to visualize (default: auto-pick top SFA rank)")
    parser.add_argument("--target_classes", nargs="+", default=["TUM", "NORM", "STR"],
                        help="Tissue classes to visualize from episode")
    parser.add_argument("--episode_idx", type=int, default=0,
                        help="Which episode to use from manifest")
    parser.add_argument("--output_dir", type=str, default="results/sfa_pathology",
                        help="Output directory for generated figures")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"🚀 Initializing SFA Pathology Visualization on device: {device}")

    # 1. Load Models
    print(f"Loading backbone from: {args.backbone}")
    backbone, conv_body = load_backbone(args.backbone, device)

    print(f"Loading KAN-ProtoNet metric from: {args.kan_metric}")
    kan_method, kan_config = load_kan_method(args.kan_metric, device)

    transform = get_eval_transform(args.image_size)

    # 2. Extract Top SFA Dimensions
    kan_linear = kan_method.kan_distance
    A_norm, A_raw = compute_sfa_scores(kan_linear, mode="global")
    sorted_dims = torch.argsort(A_raw, descending=True).cpu().numpy()

    top_dim_primary = args.top_dim if args.top_dim is not None else int(sorted_dims[0])
    top_rank_primary = int(np.where(sorted_dims == top_dim_primary)[0][0]) + 1
    top_score_primary = float(A_norm[top_dim_primary].item())

    print("\n" + "=" * 60)
    print("📊 Top 5 SFA Feature Dimensions in KAN-Distance Metric:")
    for rank, d in enumerate(sorted_dims[:5], 1):
        print(f"  Rank #{rank}: Dimension {d:3d} | SFA Score = {A_norm[d].item():.4f} (Raw = {A_raw[d].item():.4f})")
    print("=" * 60 + "\n")

    # 3. Process Standalone Image Mode
    if args.image_path is not None:
        img_p = Path(args.image_path)
        if not img_p.exists():
            raise FileNotFoundError(f"Image not found: {img_p}")
        pil_img = Image.open(img_p).convert("RGB")
        tensor_x = transform(pil_img).unsqueeze(0).to(device)
        rgb_np = denormalize(tensor_x.squeeze(0))

        with torch.no_grad():
            Z = conv_body(tensor_x)
            M_norm = synthesize_sfa_heatmap(Z, A_norm, target_size=(args.image_size, args.image_size))

        out_name = f"triptych_standalone_{img_p.stem}_dim{top_dim_primary}_{args.colormap}.png"
        out_path = os.path.join(args.output_dir, out_name)
        generate_triptych_figure(
            rgb_img=rgb_np,
            heatmap_norm=M_norm,
            kan_distance_layer=kan_linear,
            top_dim_idx=top_dim_primary,
            sfa_rank=top_rank_primary,
            sfa_score=top_score_primary,
            class_code="SAMPLE",
            output_path=out_path,
            colormap_name=args.colormap,
            alpha=args.alpha,
            draw_contour=args.contour
        )
        print("Done!")
        return

    # 4. Episode Manifest Mode
    print(f"Loading episode manifest: {args.manifest}")
    with open(args.manifest, "r", encoding="utf-8") as f:
        episodes = json.load(f)

    if args.episode_idx >= len(episodes):
        raise IndexError(f"Episode index {args.episode_idx} out of range ({len(episodes)} total)")

    ep = episodes[args.episode_idx]
    root = Path(ep["dataset_root"])
    if not root.exists() and (Path("..") / root).exists():
        root = Path("..") / root

    label_map = ep["label_map"]
    inv_label_map = {v: k for k, v in label_map.items()}
    n_way = ep["n_way"]

    # Compute Support Prototypes
    print("Computing class prototypes from support set...")
    support_imgs = []
    support_labels = []
    for cls_name, rel_paths in ep["support"].items():
        local_lbl = label_map[cls_name]
        for rp in rel_paths:
            img_path = root / rp
            pil_img = Image.open(img_path).convert("RGB")
            support_imgs.append(transform(pil_img))
            support_labels.append(local_lbl)

    x_s = torch.stack(support_imgs).unsqueeze(0).to(device)
    y_s = torch.tensor(support_labels).unsqueeze(0).to(device)

    with torch.no_grad():
        z_s = backbone(x_s.view(-1, 3, args.image_size, args.image_size)).view(1, -1, 512)
        if kan_method.kan_prenorm:
            z_s = F.normalize(z_s, p=2, dim=-1)
        centroids = compute_centroids(z_s, y_s, n_way=n_way).squeeze(0)  # [n_way, 512]

    # Process Query Images for Requested Classes
    collected_samples_data = []

    for target_cls in args.target_classes:
        if target_cls not in label_map:
            print(f"⚠️ Target class {target_cls} not in episode classes. Skipping.")
            continue

        target_lbl = label_map[target_cls]
        cls_queries = [q for q in ep["query"] if q["label"] == target_lbl]
        if not cls_queries:
            continue

        # Select first query image for this class
        selected_q = cls_queries[0]
        q_img_path = root / selected_q["path"]
        pil_img = Image.open(q_img_path).convert("RGB")
        tensor_q = transform(pil_img).unsqueeze(0).to(device)
        rgb_np = denormalize(tensor_q.squeeze(0))

        with torch.no_grad():
            Z = conv_body(tensor_q)  # [1, 512, 7, 7]
            z_q = backbone(tensor_q)  # [1, 512]
            if kan_method.kan_prenorm:
                z_q = F.normalize(z_q, p=2, dim=-1)

            # Feature difference to this class's prototype
            proto_c = centroids[target_lbl]  # [512]
            delta_primary = (z_q[0, top_dim_primary] - proto_c[top_dim_primary]).item()

            # Synthesize SFA Heatmap
            M_norm = synthesize_sfa_heatmap(Z, A_norm, target_size=(args.image_size, args.image_size))

        out_filename = f"triptych_{target_cls}_dim{top_dim_primary}_{args.colormap}.png"
        out_path = os.path.join(args.output_dir, out_filename)

        # Generate Individual Publication Triptych
        generate_triptych_figure(
            rgb_img=rgb_np,
            heatmap_norm=M_norm,
            kan_distance_layer=kan_linear,
            top_dim_idx=top_dim_primary,
            sfa_rank=top_rank_primary,
            sfa_score=top_score_primary,
            class_code=target_cls,
            output_path=out_path,
            colormap_name=args.colormap,
            alpha=args.alpha,
            query_delta=delta_primary,
            draw_contour=args.contour,
            episode_id=args.episode_idx
        )

        collected_samples_data.append({
            "rgb_img": rgb_np,
            "heatmap_norm": M_norm,
            "class_code": target_cls,
            "top_dim": top_dim_primary,
            "sfa_score": top_score_primary,
            "kan_layer": kan_linear,
            "query_delta": delta_primary
        })

    # Generate Summary Multi-Class Grid
    if len(collected_samples_data) > 1:
        grid_out = os.path.join(args.output_dir, f"sfa_pathology_multiclass_grid_{args.colormap}.png")
        generate_multiclass_grid(collected_samples_data, grid_out, colormap_name=args.colormap, alpha=args.alpha)

    print("\n🎉 All SFA Pathology Attention Maps successfully generated!")
    print(f"📁 Check output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
