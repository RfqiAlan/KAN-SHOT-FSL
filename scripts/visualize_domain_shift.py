"""
Domain-Shift Visualization Script (FHIST-style).
Generates two publication-ready figures using frozen ResNet-18 features:

1. t-SNE Intra-Dataset Class Clustering (cf. FHIST Fig. 2)
   - Separate panels for NCT-CRC (9 classes) and LC25000 (5 classes)
   - Shows whether tissue classes form distinct clusters in feature space

2. PCA Domain-Shift Plot (cf. FHIST Fig. 3)
   - NCT-CRC vs LC25000 samples overlaid in the same 2D PCA space
   - Reveals the magnitude and direction of cross-domain feature shift

Outputs are saved as high-resolution PNG and PDF in results/domain_shift/.

Usage:
    python scripts/visualize_domain_shift.py
    python scripts/visualize_domain_shift.py --backbone checkpoints/seed2021/resnet18_nct.pth
    python scripts/visualize_domain_shift.py --n_samples 3000 --output_dir results/domain_shift
"""
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Dataset Definitions ──
NCT_CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
LC_CLASSES = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]

# Prettier display names for the paper
NCT_DISPLAY = {
    "ADI": "Adipose", "BACK": "Background", "DEB": "Debris",
    "LYM": "Lymphocyte", "MUC": "Mucus", "MUS": "Muscle",
    "NORM": "Normal", "STR": "Stroma", "TUM": "Tumor"
}
LC_DISPLAY = {
    "colon_aca": "Colon ACA", "colon_n": "Colon Normal",
    "lung_aca": "Lung ACA", "lung_n": "Lung Normal", "lung_scc": "Lung SCC"
}


def load_backbone(backbone_path: str, device: torch.device) -> nn.Module:
    """Load frozen ResNet-18 feature extractor."""
    model = models.resnet18(weights=None)
    model.fc = nn.Identity()

    raw = torch.load(backbone_path, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw)
    state = {k: v for k, v in state.items()
             if not k.startswith("fc.")
             and not k.endswith(("fc.weight", "fc.bias",
                                 "classifier.weight", "classifier.bias"))}
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model


def sample_images_from_dir(data_root: Path, classes: list, n_per_class: int,
                           seed: int = 42) -> tuple:
    """
    Sample n_per_class images from each class directory.
    Returns: (image_paths, class_labels_str)
    """
    rng = random.Random(seed)
    all_paths = []
    all_labels = []

    for cls in classes:
        cls_dir = data_root / cls
        if not cls_dir.exists():
            print(f"⚠️  Class directory not found: {cls_dir}")
            continue
        imgs = sorted([p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff')])
        sampled = rng.sample(imgs, min(n_per_class, len(imgs)))
        all_paths.extend(sampled)
        all_labels.extend([cls] * len(sampled))

    return all_paths, all_labels


def extract_features_batch(model: nn.Module, image_paths: list,
                           transform, device: torch.device,
                           batch_size: int = 64) -> np.ndarray:
    """Extract 512-d features from a list of image paths in batches."""
    all_features = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert('RGB')
            imgs.append(transform(img))
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = model(batch)  # [B, 512]
        all_features.append(feats.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def setup_paper_style():
    """Configure matplotlib for Springer Nature paper quality."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def plot_tsne_classes(features: np.ndarray, labels: list, class_names: list,
                      display_map: dict, title: str, save_path: str,
                      perplexity: int = 30, seed: int = 42):
    """
    Generate t-SNE scatter plot showing intra-dataset class clustering.
    Inspired by FHIST Figure 2.
    """
    print(f"  Computing t-SNE ({features.shape[0]} samples, perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                n_iter=1000, learning_rate='auto', init='pca')
    coords = tsne.fit_transform(features)

    # Assign numerical labels
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    numeric_labels = np.array([label_to_idx[l] for l in labels])

    n_classes = len(class_names)
    palette = sns.color_palette("husl", n_classes)

    fig, ax = plt.subplots(figsize=(8, 7))

    for i, cls in enumerate(class_names):
        mask = (numeric_labels == i)
        display_name = display_map.get(cls, cls)
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[palette[i]], label=display_name,
            alpha=0.55, s=12, edgecolors='none', rasterized=True
        )

    ax.set_title(title, fontweight='bold', pad=12)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(
        loc='upper right', framealpha=0.9, edgecolor='0.8',
        markerscale=2.5, handletextpad=0.5,
        ncol=2 if n_classes > 5 else 1
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # Subtle grid
    ax.grid(True, alpha=0.15, linestyle='--')
    sns.despine(ax=ax)

    for fmt in ('png', 'pdf'):
        out = f"{save_path}.{fmt}"
        fig.savefig(out, format=fmt)
        print(f"  💾 Saved: {out}")
    plt.close(fig)


def plot_pca_domain_shift(feats_source: np.ndarray, feats_target: np.ndarray,
                          source_name: str, target_name: str,
                          title: str, save_path: str):
    """
    Generate PCA domain-shift visualization.
    Inspired by FHIST Figure 3: two domains overlaid in the same PCA space.
    """
    combined = np.concatenate([feats_source, feats_target], axis=0)
    n_source = feats_source.shape[0]

    print(f"  Computing PCA ({combined.shape[0]} samples)...")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(combined)

    source_coords = coords[:n_source]
    target_coords = coords[n_source:]

    # Compute explained variance
    ev1, ev2 = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(8, 7))

    # Source domain
    ax.scatter(
        source_coords[:, 0], source_coords[:, 1],
        c='#1f4e79', label=f"{source_name} (source)",
        alpha=0.35, s=10, edgecolors='none', rasterized=True
    )

    # Target domain
    ax.scatter(
        target_coords[:, 0], target_coords[:, 1],
        c='#c0504d', label=f"{target_name} (target)",
        alpha=0.35, s=10, edgecolors='none', rasterized=True
    )

    # Mark centroids
    src_centroid = source_coords.mean(axis=0)
    tgt_centroid = target_coords.mean(axis=0)

    ax.scatter(*src_centroid, c='#1f4e79', s=200, marker='X',
               edgecolors='white', linewidth=1.5, zorder=5)
    ax.scatter(*tgt_centroid, c='#c0504d', s=200, marker='X',
               edgecolors='white', linewidth=1.5, zorder=5)

    # Draw arrow from source centroid to target centroid
    ax.annotate(
        '', xy=tgt_centroid, xytext=src_centroid,
        arrowprops=dict(arrowstyle='->', color='0.3', lw=2,
                        connectionstyle='arc3,rad=0.1')
    )

    # Distance annotation
    shift_dist = np.linalg.norm(tgt_centroid - src_centroid)
    mid = (src_centroid + tgt_centroid) / 2
    ax.text(
        mid[0], mid[1] + 1.5, f"Δ = {shift_dist:.1f}",
        ha='center', fontsize=10, fontstyle='italic', color='0.3',
        path_effects=[pe.withStroke(linewidth=3, foreground='white')]
    )

    ax.set_title(title, fontweight='bold', pad=12)
    ax.set_xlabel(f"PC1 ({ev1:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({ev2:.1f}% variance)")
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='0.8',
              markerscale=2.5)

    ax.grid(True, alpha=0.15, linestyle='--')
    sns.despine(ax=ax)

    for fmt in ('png', 'pdf'):
        out = f"{save_path}.{fmt}"
        fig.savefig(out, format=fmt)
        print(f"  💾 Saved: {out}")
    plt.close(fig)


def plot_pca_domain_shift_per_class(feats_source: np.ndarray, labels_source: list,
                                     feats_target: np.ndarray, labels_target: list,
                                     source_classes: list, target_classes: list,
                                     source_display: dict, target_display: dict,
                                     title: str, save_path: str):
    """
    PCA with per-class coloring for both domains.
    Source classes use circle markers, target classes use triangle markers.
    """
    combined = np.concatenate([feats_source, feats_target], axis=0)
    n_source = feats_source.shape[0]

    print(f"  Computing PCA per-class ({combined.shape[0]} samples)...")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(combined)

    source_coords = coords[:n_source]
    target_coords = coords[n_source:]

    ev1, ev2 = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(10, 8))

    # Source classes (circles, blue palette)
    src_palette = sns.color_palette("Blues_d", len(source_classes))
    src_label_idx = {c: i for i, c in enumerate(source_classes)}
    for i, cls in enumerate(source_classes):
        mask = np.array([l == cls for l in labels_source])
        if mask.sum() == 0:
            continue
        ax.scatter(
            source_coords[mask, 0], source_coords[mask, 1],
            c=[src_palette[i]], marker='o', alpha=0.4, s=10,
            label=f"NCT: {source_display.get(cls, cls)}", edgecolors='none',
            rasterized=True
        )

    # Target classes (triangles, red palette)
    tgt_palette = sns.color_palette("Reds_d", len(target_classes))
    for i, cls in enumerate(target_classes):
        mask = np.array([l == cls for l in labels_target])
        if mask.sum() == 0:
            continue
        ax.scatter(
            target_coords[mask, 0], target_coords[mask, 1],
            c=[tgt_palette[i]], marker='^', alpha=0.4, s=14,
            label=f"LC: {target_display.get(cls, cls)}", edgecolors='none',
            rasterized=True
        )

    ax.set_title(title, fontweight='bold', pad=12)
    ax.set_xlabel(f"PC1 ({ev1:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({ev2:.1f}% variance)")
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='0.8',
              markerscale=2.5, ncol=2, fontsize=8)
    ax.grid(True, alpha=0.15, linestyle='--')
    sns.despine(ax=ax)

    for fmt in ('png', 'pdf'):
        out = f"{save_path}.{fmt}"
        fig.savefig(out, format=fmt)
        print(f"  💾 Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Domain-Shift Visualization (FHIST-style)"
    )
    parser.add_argument(
        "--backbone", type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "seed2021" / "resnet18_nct.pth"),
        help="Path to frozen ResNet-18 backbone"
    )
    parser.add_argument(
        "--nct_root", type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "nct_train"),
        help="Path to NCT-CRC training data (9 class folders)"
    )
    parser.add_argument(
        "--lc_root", type=str,
        default=str(PROJECT_ROOT / "data" / "lc25000_flat"),
        help="Path to LC25000 flat data (5 class folders)"
    )
    parser.add_argument("--n_samples", type=int, default=5000,
                        help="Total samples per domain (split across classes)")
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--tsne_perplexity", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="results/domain_shift")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_paper_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load Backbone ──
    print(f"🔧 Loading backbone: {args.backbone}")
    model = load_backbone(args.backbone, device)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ── Sample & Extract Features: NCT-CRC ──
    n_per_class_nct = args.n_samples // len(NCT_CLASSES)
    print(f"\n📷 Sampling {n_per_class_nct} images/class from NCT-CRC ({len(NCT_CLASSES)} classes)...")
    nct_paths, nct_labels = sample_images_from_dir(
        Path(args.nct_root), NCT_CLASSES, n_per_class_nct, seed=args.seed
    )
    print(f"   Total NCT samples: {len(nct_paths)}")
    nct_feats = extract_features_batch(model, nct_paths, transform, device, args.batch_size)

    # ── Sample & Extract Features: LC25000 ──
    n_per_class_lc = args.n_samples // len(LC_CLASSES)
    print(f"\n📷 Sampling {n_per_class_lc} images/class from LC25000 ({len(LC_CLASSES)} classes)...")
    lc_paths, lc_labels = sample_images_from_dir(
        Path(args.lc_root), LC_CLASSES, n_per_class_lc, seed=args.seed
    )
    print(f"   Total LC25000 samples: {len(lc_paths)}")
    lc_feats = extract_features_batch(model, lc_paths, transform, device, args.batch_size)

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 1: t-SNE Intra-Dataset Class Clustering (FHIST Fig. 2 style)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("🎨 Figure 1: t-SNE Class Clustering")
    print(f"{'=' * 60}")

    # NCT-CRC t-SNE
    print("\n[NCT-CRC-HE-100K]")
    plot_tsne_classes(
        nct_feats, nct_labels, NCT_CLASSES, NCT_DISPLAY,
        title="t-SNE Feature Embedding: NCT-CRC-HE (9 Classes)",
        save_path=os.path.join(args.output_dir, "tsne_nct_classes"),
        perplexity=args.tsne_perplexity, seed=args.seed
    )

    # LC25000 t-SNE
    print("\n[LC25000]")
    plot_tsne_classes(
        lc_feats, lc_labels, LC_CLASSES, LC_DISPLAY,
        title="t-SNE Feature Embedding: LC25000 (5 Classes)",
        save_path=os.path.join(args.output_dir, "tsne_lc_classes"),
        perplexity=args.tsne_perplexity, seed=args.seed
    )

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 2: PCA Domain-Shift (FHIST Fig. 3 style)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("🎨 Figure 2: PCA Domain-Shift Visualization")
    print(f"{'=' * 60}")

    # Simple two-color domain overlay
    plot_pca_domain_shift(
        nct_feats, lc_feats,
        source_name="NCT-CRC (Source)",
        target_name="LC25000 (Target)",
        title="PCA Domain Shift: NCT-CRC → LC25000",
        save_path=os.path.join(args.output_dir, "pca_domain_shift")
    )

    # Per-class domain overlay (bonus, more detailed)
    plot_pca_domain_shift_per_class(
        nct_feats, nct_labels, lc_feats, lc_labels,
        NCT_CLASSES, LC_CLASSES, NCT_DISPLAY, LC_DISPLAY,
        title="PCA Domain Shift (Per-Class): NCT-CRC → LC25000",
        save_path=os.path.join(args.output_dir, "pca_domain_shift_perclass")
    )

    # ── Feature Statistics ──
    print(f"\n{'=' * 60}")
    print("📊 Feature Space Statistics")
    print(f"{'=' * 60}")

    nct_mean = nct_feats.mean(axis=0)
    lc_mean = lc_feats.mean(axis=0)
    l2_shift = np.linalg.norm(nct_mean - lc_mean)
    cosine_sim = np.dot(nct_mean, lc_mean) / (np.linalg.norm(nct_mean) * np.linalg.norm(lc_mean))

    print(f"   NCT-CRC feature mean norm:  {np.linalg.norm(nct_mean):.4f}")
    print(f"   LC25000 feature mean norm:   {np.linalg.norm(lc_mean):.4f}")
    print(f"   L2 domain shift (centroids): {l2_shift:.4f}")
    print(f"   Cosine similarity:           {cosine_sim:.4f}")
    print(f"   NCT feature std (avg):       {nct_feats.std(axis=0).mean():.4f}")
    print(f"   LC  feature std (avg):       {lc_feats.std(axis=0).mean():.4f}")

    # Save stats
    stats_path = os.path.join(args.output_dir, "domain_shift_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Domain Shift Statistics\n")
        f.write(f"Backbone: {args.backbone}\n")
        f.write(f"NCT samples: {len(nct_paths)}\n")
        f.write(f"LC  samples: {len(lc_paths)}\n")
        f.write(f"Feature dim: {nct_feats.shape[1]}\n")
        f.write(f"L2 shift:    {l2_shift:.4f}\n")
        f.write(f"Cosine sim:  {cosine_sim:.4f}\n")
    print(f"   💾 Saved stats: {stats_path}")

    print(f"\n✅ All visualizations saved to {args.output_dir}/")
    print("   Files generated:")
    for f in sorted(Path(args.output_dir).iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"   • {f.name} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
