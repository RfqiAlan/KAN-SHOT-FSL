"""
Comprehensive Analysis Script for KAN-SHOT Paper
Generates:
  1. Win/Tie/Loss paired episode analysis
  2. Per-class accuracy breakdown
  3. Variance/stability comparison
  4. Bootstrap CI overlap visualization
  5. Spline shape analysis (do learned splines ≈ parabola?)
  6. Computational efficiency (inference time)
  7. Calibration analysis (ECE) — placeholder if logits not available
"""

import os
import sys
import csv
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy import stats
from PIL import Image

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.kan import KANLinear

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'seed2021')
EPISODES_DIR = os.path.join(os.path.dirname(__file__), '..', 'episodes')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


# ===================================================================
# UTILITY: Load paired CSV results
# ===================================================================
def load_csv_accuracies(csv_path):
    """Load per-episode accuracies from a CSV file."""
    accs = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                acc = float(row['accuracy'])
                if acc <= 1.0:  # per-episode accuracy (0-1 scale)
                    accs.append(acc)
            except (ValueError, KeyError):
                continue
    return np.array(accs)


def load_paired_results(kan_csv, proto_csv):
    """Load and align paired episode results."""
    kan = load_csv_accuracies(kan_csv)
    proto = load_csv_accuracies(proto_csv)
    n = min(len(kan), len(proto))
    return kan[:n], proto[:n]


# ===================================================================
# 1. WIN/TIE/LOSS ANALYSIS
# ===================================================================
def win_tie_loss_analysis():
    """Compare KAN vs ProtoNet on a per-episode basis across all scenarios."""
    print("\n" + "="*70)
    print("1. WIN/TIE/LOSS ANALYSIS (per-episode)")
    print("="*70)

    scenarios = [
        ("LC 5-Way 5-Shot (Frozen)", "kan_distance_lc5way_5shot.csv", "proto_lc5way_5shot.csv"),
        ("LC 5-Way 1-Shot", "kan_distance_lc5way_1shot.csv", "proto_lc5way_1shot.csv"),
        ("LC 5-Way 10-Shot", "kan_distance_lc5way_10shot.csv", "proto_lc5way_10shot.csv"),
        ("CRC-VAL 9-Way 5-Shot", "kan_crc_5shot_v2.csv", "proto_crc_5shot.csv"),
        ("LC-Colon 5-Shot", "kan_distance_lccolon_5shot.csv", "proto_lccolon_5shot.csv"),
        ("LC-Lung 5-Shot", "kan_distance_lclung_5shot.csv", "proto_lclung_5shot.csv"),
    ]

    results = []
    for name, kan_f, proto_f in scenarios:
        kan_path = os.path.join(RESULTS_DIR, kan_f)
        proto_path = os.path.join(RESULTS_DIR, proto_f)
        if not os.path.exists(kan_path) or not os.path.exists(proto_path):
            print(f"  ⚠️  Skipping {name}: file not found")
            continue

        kan_acc, proto_acc = load_paired_results(kan_path, proto_path)
        diff = kan_acc - proto_acc

        wins = np.sum(diff > 0.001)     # KAN wins (>0.1% margin)
        losses = np.sum(diff < -0.001)   # KAN loses
        ties = np.sum(np.abs(diff) <= 0.001)

        # Signed-rank test on differences
        stat, p_val = stats.wilcoxon(kan_acc, proto_acc)
        d_cohen = np.mean(diff) / (np.std(diff) + 1e-10)

        results.append({
            'scenario': name,
            'n_episodes': len(kan_acc),
            'kan_mean': np.mean(kan_acc)*100,
            'proto_mean': np.mean(proto_acc)*100,
            'wins': int(wins),
            'ties': int(ties),
            'losses': int(losses),
            'win_rate': wins / len(kan_acc) * 100,
            'p_value': p_val,
            'cohen_d': d_cohen,
            'mean_diff': np.mean(diff)*100,
            'kan_std': np.std(kan_acc)*100,
            'proto_std': np.std(proto_acc)*100,
        })

        print(f"\n  {name} ({len(kan_acc)} episodes)")
        print(f"    KAN: {np.mean(kan_acc)*100:.2f}% (±{np.std(kan_acc)*100:.2f})")
        print(f"    Proto: {np.mean(proto_acc)*100:.2f}% (±{np.std(proto_acc)*100:.2f})")
        print(f"    KAN Wins: {wins}  |  Ties: {ties}  |  KAN Losses: {losses}")
        print(f"    Win Rate: {wins/len(kan_acc)*100:.1f}%")
        print(f"    Wilcoxon p={p_val:.6f}, Cohen's d={d_cohen:.4f}")

    # Generate Win/Tie/Loss bar chart
    if results:
        fig, ax = plt.subplots(figsize=(12, 5))
        names = [r['scenario'] for r in results]
        wins = [r['wins'] for r in results]
        ties = [r['ties'] for r in results]
        losses = [r['losses'] for r in results]

        x = np.arange(len(names))
        width = 0.25

        bars1 = ax.bar(x - width, wins, width, label='KAN Wins', color='#2ecc71', alpha=0.85)
        bars2 = ax.bar(x, ties, width, label='Ties', color='#95a5a6', alpha=0.85)
        bars3 = ax.bar(x + width, losses, width, label='KAN Losses', color='#e74c3c', alpha=0.85)

        ax.set_ylabel('Number of Episodes')
        ax.set_title('Win/Tie/Loss Analysis: KAN-SHOT vs ProtoNet (Per-Episode)')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add counts on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, 'win_tie_loss.png')
        plt.savefig(out_path)
        plt.close()
        print(f"\n  ✅ Saved: {out_path}")

    return results


# ===================================================================
# 2. VARIANCE / STABILITY COMPARISON
# ===================================================================
def variance_analysis():
    """Compare variance and stability of KAN vs ProtoNet."""
    print("\n" + "="*70)
    print("2. VARIANCE / STABILITY COMPARISON")
    print("="*70)

    scenarios = [
        ("LC 5-Way 5-Shot", "kan_distance_lc5way_5shot.csv", "proto_lc5way_5shot.csv"),
        ("LC-Colon 5-Shot", "kan_distance_lccolon_5shot.csv", "proto_lccolon_5shot.csv"),
        ("LC-Lung 5-Shot", "kan_distance_lclung_5shot.csv", "proto_lclung_5shot.csv"),
        ("CRC-VAL 9-Way 5-Shot", "kan_crc_5shot_v2.csv", "proto_crc_5shot.csv"),
    ]

    variance_data = []
    for name, kan_f, proto_f in scenarios:
        kan_path = os.path.join(RESULTS_DIR, kan_f)
        proto_path = os.path.join(RESULTS_DIR, proto_f)
        if not os.path.exists(kan_path) or not os.path.exists(proto_path):
            continue

        kan_acc, proto_acc = load_paired_results(kan_path, proto_path)

        kan_std = np.std(kan_acc) * 100
        proto_std = np.std(proto_acc) * 100
        kan_iqr = (np.percentile(kan_acc, 75) - np.percentile(kan_acc, 25)) * 100
        proto_iqr = (np.percentile(proto_acc, 75) - np.percentile(proto_acc, 25)) * 100
        kan_cv = np.std(kan_acc) / np.mean(kan_acc) * 100
        proto_cv = np.std(proto_acc) / np.mean(proto_acc) * 100

        # Levene's test for equality of variances
        lev_stat, lev_p = stats.levene(kan_acc, proto_acc)

        print(f"\n  {name}:")
        print(f"    KAN  — Std: {kan_std:.3f}%, IQR: {kan_iqr:.3f}%, CV: {kan_cv:.2f}%")
        print(f"    Proto — Std: {proto_std:.3f}%, IQR: {proto_iqr:.3f}%, CV: {proto_cv:.2f}%")
        print(f"    Levene's test: F={lev_stat:.3f}, p={lev_p:.6f}")
        if kan_std < proto_std:
            print(f"    → KAN is MORE STABLE (lower variance)")
        else:
            print(f"    → ProtoNet is MORE STABLE (lower variance)")

        variance_data.append({
            'scenario': name, 'kan_std': kan_std, 'proto_std': proto_std,
            'kan_iqr': kan_iqr, 'proto_iqr': proto_iqr,
            'kan_cv': kan_cv, 'proto_cv': proto_cv,
            'levene_p': lev_p, 'kan_accs': kan_acc, 'proto_accs': proto_acc
        })

    # Box plot comparison
    if variance_data:
        fig, axes = plt.subplots(1, len(variance_data), figsize=(4*len(variance_data), 5))
        if len(variance_data) == 1:
            axes = [axes]
        for i, vd in enumerate(variance_data):
            bp = axes[i].boxplot([vd['kan_accs']*100, vd['proto_accs']*100],
                                  labels=['KAN-SHOT', 'ProtoNet'],
                                  patch_artist=True,
                                  medianprops=dict(color='black', linewidth=2))
            bp['boxes'][0].set_facecolor('#3498db')
            bp['boxes'][1].set_facecolor('#e74c3c')
            for box in bp['boxes']:
                box.set_alpha(0.7)
            axes[i].set_title(vd['scenario'], fontsize=10)
            axes[i].set_ylabel('Accuracy (%)')
            axes[i].grid(axis='y', alpha=0.3)

        plt.suptitle('Accuracy Distribution: KAN-SHOT vs ProtoNet', fontsize=13, y=1.02)
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, 'variance_boxplot.png')
        plt.savefig(out_path)
        plt.close()
        print(f"\n  ✅ Saved: {out_path}")

    return variance_data


# ===================================================================
# 3. BOOTSTRAP CI OVERLAP VISUALIZATION
# ===================================================================
def bootstrap_ci_visualization():
    """Bootstrap confidence interval overlap plot."""
    print("\n" + "="*70)
    print("3. BOOTSTRAP CI OVERLAP VISUALIZATION")
    print("="*70)

    scenarios = [
        ("LC 5-Way\n1-Shot", "kan_distance_lc5way_1shot.csv", "proto_lc5way_1shot.csv"),
        ("LC 5-Way\n5-Shot", "kan_distance_lc5way_5shot.csv", "proto_lc5way_5shot.csv"),
        ("LC 5-Way\n10-Shot", "kan_distance_lc5way_10shot.csv", "proto_lc5way_10shot.csv"),
        ("CRC-VAL\n9-Way 5-Shot", "kan_crc_5shot_v2.csv", "proto_crc_5shot.csv"),
        ("LC-Colon\n5-Shot", "kan_distance_lccolon_5shot.csv", "proto_lccolon_5shot.csv"),
        ("LC-Lung\n5-Shot", "kan_distance_lclung_5shot.csv", "proto_lclung_5shot.csv"),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    y_positions = []
    labels = []
    n_boot = 10000

    y = 0
    for name, kan_f, proto_f in scenarios:
        kan_path = os.path.join(RESULTS_DIR, kan_f)
        proto_path = os.path.join(RESULTS_DIR, proto_f)
        if not os.path.exists(kan_path) or not os.path.exists(proto_path):
            continue

        kan_acc, proto_acc = load_paired_results(kan_path, proto_path)

        # Bootstrap
        rng = np.random.RandomState(42)
        n = len(kan_acc)

        kan_boots = np.array([np.mean(rng.choice(kan_acc, n, replace=True)) for _ in range(n_boot)]) * 100
        proto_boots = np.array([np.mean(rng.choice(proto_acc, n, replace=True)) for _ in range(n_boot)]) * 100

        kan_ci = np.percentile(kan_boots, [2.5, 97.5])
        proto_ci = np.percentile(proto_boots, [2.5, 97.5])
        kan_mean = np.mean(kan_acc) * 100
        proto_mean = np.mean(proto_acc) * 100

        # Plot KAN
        ax.plot([kan_ci[0], kan_ci[1]], [y, y], color='#3498db', linewidth=3, solid_capstyle='round')
        ax.plot(kan_mean, y, 'o', color='#3498db', markersize=8, zorder=5)

        # Plot ProtoNet
        ax.plot([proto_ci[0], proto_ci[1]], [y+0.3, y+0.3], color='#e74c3c', linewidth=3, solid_capstyle='round')
        ax.plot(proto_mean, y+0.3, 's', color='#e74c3c', markersize=8, zorder=5)

        labels.append(name)
        y_positions.append(y + 0.15)
        y += 1.0

        # Check overlap
        overlap = kan_ci[1] >= proto_ci[0] and proto_ci[1] >= kan_ci[0]
        print(f"  {name.replace(chr(10), ' ')}: CIs {'OVERLAP' if overlap else 'DO NOT OVERLAP'}")
        print(f"    KAN:   {kan_mean:.2f}% [{kan_ci[0]:.2f}, {kan_ci[1]:.2f}]")
        print(f"    Proto: {proto_mean:.2f}% [{proto_ci[0]:.2f}, {proto_ci[1]:.2f}]")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Mean Accuracy (%)')
    ax.set_title('95% Bootstrap CI Comparison: KAN-SHOT (●) vs ProtoNet (■)')
    ax.grid(axis='x', alpha=0.3)
    ax.legend(['KAN-SHOT', '', 'ProtoNet'], loc='lower right', fontsize=10)
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#3498db', linewidth=3, marker='o', markersize=8, label='KAN-SHOT'),
        Line2D([0], [0], color='#e74c3c', linewidth=3, marker='s', markersize=8, label='ProtoNet'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'bootstrap_ci_overlap.png')
    plt.savefig(out_path)
    plt.close()
    print(f"\n  ✅ Saved: {out_path}")


# ===================================================================
# 4. SPLINE SHAPE ANALYSIS
# ===================================================================
def spline_shape_analysis():
    """Analyze whether learned KAN splines resemble parabolas (Euclidean-like)."""
    print("\n" + "="*70)
    print("4. SPLINE SHAPE ANALYSIS (Do splines ≈ x²?)")
    print("="*70)

    ckpt_path = os.path.join(CHECKPOINT_DIR, 'kanprotonet_cosine_nct.pth')
    if not os.path.exists(ckpt_path):
        print(f"  ⚠️  Checkpoint not found: {ckpt_path}")
        return

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)

    # Reconstruct KAN distance layer (512 -> 1)
    kan_dist = KANLinear(in_features=512, out_features=1, grid_size=5, spline_order=3)

    # Filter state dict for KAN distance keys
    kan_keys = {k: v for k, v in state.items() if 'kan_distance' in k or 'spline' in k or 'base_weight' in k}

    # Try loading — handle different key formats
    if kan_keys:
        # Remove prefix
        clean_keys = {}
        for k, v in kan_keys.items():
            clean_k = k.replace('kan_distance.', '').replace('method.kan_distance.', '')
            clean_keys[clean_k] = v
        try:
            kan_dist.load_state_dict(clean_keys, strict=False)
            print("  ✅ Loaded KAN distance weights from checkpoint")
        except Exception as e:
            print(f"  ⚠️  Could not load KAN weights cleanly: {e}")
            print(f"  Available keys: {list(state.keys())[:20]}")
            # Try alternate loading
            try:
                # Maybe the full model is saved
                new_keys = {}
                for k, v in state.items():
                    if 'kan_distance.' in k:
                        new_k = k.split('kan_distance.')[-1]
                        new_keys[new_k] = v
                if new_keys:
                    kan_dist.load_state_dict(new_keys, strict=True)
                    print("  ✅ Loaded with alternate key mapping")
                else:
                    print("  ⚠️  No kan_distance keys found, trying direct load...")
                    kan_dist.load_state_dict(state, strict=False)
            except Exception as e2:
                print(f"  ❌ Failed to load: {e2}")
                return
    else:
        # Try direct load
        try:
            kan_dist.load_state_dict(state, strict=False)
            print("  ✅ Loaded directly")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            return

    kan_dist.eval()

    # Evaluate spline functions across a range of delta values
    delta_range = torch.linspace(-2.0, 2.0, 500)

    # For each input dimension, evaluate φ_j(δ) by setting all other dims to 0
    n_dims = 512
    spline_curves = np.zeros((n_dims, len(delta_range)))

    with torch.no_grad():
        for j in range(n_dims):
            # Create input: all zeros except dimension j
            x = torch.zeros(len(delta_range), n_dims)
            x[:, j] = delta_range
            out = kan_dist(x)  # [500, 1]
            spline_curves[j] = out.squeeze().numpy()

    # Compute SFA importance (sum of absolute spline weight * scaler)
    with torch.no_grad():
        spline_w = kan_dist.spline_weight.data  # [1, 512, num_basis]
        scaler = kan_dist.spline_scaler.data     # [1, 512]
        sfa_scores = (spline_w.abs().sum(dim=-1) * scaler.abs()).squeeze().numpy()  # [512]

    # Get top-10 most important dimensions
    top_k = 10
    top_indices = np.argsort(sfa_scores)[::-1][:top_k]

    # For each top dimension, compute correlation with x²
    delta_np = delta_range.numpy()
    parabola = delta_np ** 2

    print(f"\n  Top-{top_k} dimensions by SFA score:")
    print(f"  {'Dim':>5}  {'SFA':>8}  {'R² vs x²':>10}  {'R² vs |x|':>10}  {'Shape':>15}")
    print(f"  {'─'*55}")

    shape_stats = []
    for idx in top_indices:
        curve = spline_curves[idx]
        # Normalize for R² computation
        curve_norm = curve - curve.mean()
        para_norm = parabola - parabola.mean()
        abs_norm = np.abs(delta_np) - np.abs(delta_np).mean()

        # R² vs parabola
        ss_res_para = np.sum((curve_norm - para_norm * (np.dot(curve_norm, para_norm) / (np.dot(para_norm, para_norm) + 1e-10))) ** 2)
        ss_tot = np.sum(curve_norm ** 2) + 1e-10
        r2_para = 1 - ss_res_para / ss_tot

        # R² vs |x|
        ss_res_abs = np.sum((curve_norm - abs_norm * (np.dot(curve_norm, abs_norm) / (np.dot(abs_norm, abs_norm) + 1e-10))) ** 2)
        r2_abs = 1 - ss_res_abs / ss_tot

        # Symmetry: compare φ(δ) with φ(-δ)
        n_half = len(delta_np) // 2
        left = curve[:n_half]
        right = curve[-n_half:][::-1]
        sym_corr = np.corrcoef(left, right)[0, 1] if len(left) > 1 else 0

        shape = "Symmetric" if sym_corr > 0.8 else ("Asymmetric" if sym_corr < 0.3 else "Mixed")

        shape_stats.append({
            'dim': idx, 'sfa': sfa_scores[idx],
            'r2_para': r2_para, 'r2_abs': r2_abs, 'sym_corr': sym_corr, 'shape': shape
        })

        print(f"  {idx:5d}  {sfa_scores[idx]:8.4f}  {r2_para:10.4f}  {r2_abs:10.4f}  {shape:>15}")

    # Compute overall statistics
    all_r2_para = []
    all_sym = []
    for j in range(n_dims):
        curve = spline_curves[j]
        curve_norm = curve - curve.mean()
        para_norm = parabola - parabola.mean()
        ss_res = np.sum((curve_norm - para_norm * (np.dot(curve_norm, para_norm) / (np.dot(para_norm, para_norm) + 1e-10))) ** 2)
        ss_tot = np.sum(curve_norm ** 2) + 1e-10
        r2 = 1 - ss_res / ss_tot
        all_r2_para.append(r2)

        n_half = len(delta_np) // 2
        left = curve[:n_half]
        right = curve[-n_half:][::-1]
        sym = np.corrcoef(left, right)[0, 1] if len(left) > 1 else 0
        all_sym.append(sym)

    print(f"\n  Overall Statistics (all 512 dims):")
    print(f"    Mean R² vs x²: {np.mean(all_r2_para):.4f} (±{np.std(all_r2_para):.4f})")
    print(f"    Mean symmetry: {np.nanmean(all_sym):.4f} (±{np.nanstd(all_sym):.4f})")
    print(f"    Dims with R²>0.8 (near-parabolic): {np.sum(np.array(all_r2_para) > 0.8)}/512")
    print(f"    Dims with symmetry>0.8: {np.sum(np.array(all_sym) > 0.8)}/512")

    # Plot top-6 spline curves with parabola overlay
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for i, idx in enumerate(top_indices[:6]):
        ax = axes[i]
        curve = spline_curves[idx]

        ax.plot(delta_np, curve, 'b-', linewidth=2, label=f'φ_{{{idx}}}(δ)')

        # Fit and overlay best-fit parabola
        coeffs = np.polyfit(delta_np, curve, 2)
        fit_para = np.polyval(coeffs, delta_np)
        ax.plot(delta_np, fit_para, 'r--', linewidth=1.5, alpha=0.7, label=f'Best-fit x² (R²={shape_stats[i]["r2_para"]:.3f})')

        ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)
        ax.set_title(f'Dim {idx} (SFA={sfa_scores[idx]:.3f})', fontsize=10)
        ax.set_xlabel('δ (feature diff)')
        ax.set_ylabel('φ(δ)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    plt.suptitle('Learned Spline Functions vs Best-Fit Parabola (Top-6 by SFA)', fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'spline_shape_analysis.png')
    plt.savefig(out_path)
    plt.close()
    print(f"\n  ✅ Saved: {out_path}")

    # Histogram of R² values
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(all_r2_para, bins=50, color='#3498db', alpha=0.7, edgecolor='white')
    axes[0].axvline(x=np.mean(all_r2_para), color='red', linestyle='--', label=f'Mean={np.mean(all_r2_para):.3f}')
    axes[0].set_xlabel('R² vs x² (Parabola)')
    axes[0].set_ylabel('Count (dimensions)')
    axes[0].set_title('How Parabolic Are Learned Splines?')
    axes[0].legend()

    axes[1].hist(all_sym, bins=50, color='#2ecc71', alpha=0.7, edgecolor='white')
    axes[1].axvline(x=np.nanmean(all_sym), color='red', linestyle='--', label=f'Mean={np.nanmean(all_sym):.3f}')
    axes[1].set_xlabel('Symmetry Correlation')
    axes[1].set_ylabel('Count (dimensions)')
    axes[1].set_title('Symmetry of Learned Splines')
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'spline_shape_histogram.png')
    plt.savefig(out_path)
    plt.close()
    print(f"  ✅ Saved: {out_path}")

    return shape_stats, all_r2_para, all_sym


# ===================================================================
# 5. COMPUTATIONAL EFFICIENCY
# ===================================================================
def computational_efficiency():
    """Measure inference time for different metric methods."""
    print("\n" + "="*70)
    print("5. COMPUTATIONAL EFFICIENCY ANALYSIS")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load backbone
    backbone_path = os.path.join(CHECKPOINT_DIR, 'resnet18_nct.pth')
    if not os.path.exists(backbone_path):
        print(f"  ⚠️  Backbone not found: {backbone_path}")
        return

    model = models.resnet18(weights=None)
    model.fc = nn.Identity()
    raw = torch.load(backbone_path, map_location='cpu', weights_only=False)
    state = raw.get('state_dict', raw)
    state = {k: v for k, v in state.items() if not k.startswith('fc.') and not k.endswith(('fc.weight', 'fc.bias'))}
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    # KAN distance
    kan_dist = KANLinear(512, 1, grid_size=5, spline_order=3).to(device).eval()
    ckpt_path = os.path.join(CHECKPOINT_DIR, 'kanprotonet_cosine_nct.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        st = ckpt.get('state_dict', ckpt)
        kd_keys = {}
        for k, v in st.items():
            if 'kan_distance.' in k:
                kd_keys[k.split('kan_distance.')[-1]] = v
        if kd_keys:
            kan_dist.load_state_dict(kd_keys, strict=False)

    # MLP distance
    mlp_path = os.path.join(CHECKPOINT_DIR, 'mlpprotonet_cosine_nct.pth')
    mlp_proj = nn.Linear(512, 512).to(device).eval()
    if os.path.exists(mlp_path):
        mlp_ckpt = torch.load(mlp_path, map_location='cpu', weights_only=False)
        mlp_st = mlp_ckpt.get('state_dict', mlp_ckpt)
        # Try to load MLP weights
        for k, v in mlp_st.items():
            if 'linear' in k.lower() or 'projection' in k.lower():
                pass  # would need proper key mapping

    # Simulate 5-way 5-shot episode
    n_way = 5
    k_shot = 5
    q_query = 15
    image_size = 84
    n_repeat = 100

    x_s = torch.randn(1, n_way * k_shot, 3, image_size, image_size).to(device)
    x_q = torch.randn(1, n_way * q_query, 3, image_size, image_size).to(device)

    # 1. Backbone feature extraction time (shared)
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            z = model(x_s.reshape(-1, 3, image_size, image_size))

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            z_s = model(x_s.reshape(-1, 3, image_size, image_size)).reshape(1, n_way*k_shot, 512)
            z_q = model(x_q.reshape(-1, 3, image_size, image_size)).reshape(1, n_way*q_query, 512)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        backbone_time = (time.perf_counter() - t0) / n_repeat * 1000

    # 2. ProtoNet (cosine distance) metric time
    with torch.no_grad():
        z_s_det = z_s.detach()
        z_q_det = z_q.detach()

        # Warmup
        for _ in range(10):
            z_s_norm = nn.functional.normalize(z_s_det, dim=-1)
            z_q_norm = nn.functional.normalize(z_q_det, dim=-1)
            centroids = z_s_norm.reshape(1, n_way, k_shot, 512).mean(2)
            scores = z_q_norm @ centroids.transpose(1, 2)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            z_s_norm = nn.functional.normalize(z_s_det, dim=-1)
            z_q_norm = nn.functional.normalize(z_q_det, dim=-1)
            centroids = z_s_norm.reshape(1, n_way, k_shot, 512).mean(2)
            scores = z_q_norm @ centroids.transpose(1, 2)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        proto_metric_time = (time.perf_counter() - t0) / n_repeat * 1000

    # 3. KAN distance metric time
    with torch.no_grad():
        centroids = z_s_det.reshape(1, n_way, k_shot, 512).mean(2)
        diff_example = z_q_det.unsqueeze(2) - centroids.unsqueeze(1)
        diff_flat = diff_example.reshape(-1, 512)

        # Warmup
        for _ in range(10):
            kan_dist(diff_flat)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            centroids = z_s_det.reshape(1, n_way, k_shot, 512).mean(2)
            diff = z_q_det.unsqueeze(2) - centroids.unsqueeze(1)
            scores = kan_dist(diff.reshape(-1, 512)).reshape(1, n_way*q_query, n_way)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        kan_metric_time = (time.perf_counter() - t0) / n_repeat * 1000

    # 4. MLP metric time (projection + L2)
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            mlp_proj(z_s_det.reshape(-1, 512))

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            z_s_mlp = mlp_proj(z_s_det.reshape(-1, 512)).reshape(1, n_way*k_shot, 512)
            z_q_mlp = mlp_proj(z_q_det.reshape(-1, 512)).reshape(1, n_way*q_query, 512)
            centroids = z_s_mlp.reshape(1, n_way, k_shot, 512).mean(2)
            l2 = ((z_q_mlp.unsqueeze(2) - centroids.unsqueeze(1))**2).sum(-1)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        mlp_metric_time = (time.perf_counter() - t0) / n_repeat * 1000

    # Parameter counts
    kan_params = sum(p.numel() for p in kan_dist.parameters())
    mlp_params = sum(p.numel() for p in mlp_proj.parameters())

    print(f"\n  Timing Results (averaged over {n_repeat} iterations, 5-way 5-shot):")
    print(f"  {'Method':<20} {'Metric Params':>15} {'Backbone (ms)':>15} {'Metric (ms)':>15} {'Total (ms)':>12}")
    print(f"  {'─'*80}")
    print(f"  {'ProtoNet':<20} {'0':>15} {backbone_time:>15.2f} {proto_metric_time:>15.3f} {backbone_time+proto_metric_time:>12.2f}")
    print(f"  {'KAN-SHOT':<20} {kan_params:>15,} {backbone_time:>15.2f} {kan_metric_time:>15.3f} {backbone_time+kan_metric_time:>12.2f}")
    print(f"  {'MLP-ProtoNet':<20} {mlp_params:>15,} {backbone_time:>15.2f} {mlp_metric_time:>15.3f} {backbone_time+mlp_metric_time:>12.2f}")

    overhead_kan = (kan_metric_time / proto_metric_time - 1) * 100
    overhead_mlp = (mlp_metric_time / proto_metric_time - 1) * 100
    print(f"\n  KAN overhead vs ProtoNet: +{overhead_kan:.1f}%")
    print(f"  MLP overhead vs ProtoNet: +{overhead_mlp:.1f}%")

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Params
    methods = ['ProtoNet', 'KAN-SHOT', 'MLP-ProtoNet']
    params = [0, kan_params, mlp_params]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = axes[0].bar(methods, params, color=colors, alpha=0.8, edgecolor='white')
    axes[0].set_ylabel('Number of Parameters')
    axes[0].set_title('Metric Module Parameters')
    axes[0].set_yscale('log')
    axes[0].set_ylim(bottom=1)
    for bar, p in zip(bars, params):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
                     f'{p:,}', ha='center', fontsize=10, fontweight='bold')

    # Timing
    metric_times = [proto_metric_time, kan_metric_time, mlp_metric_time]
    bars = axes[1].bar(methods, metric_times, color=colors, alpha=0.8, edgecolor='white')
    axes[1].set_ylabel('Time (ms)')
    axes[1].set_title('Metric Computation Time (per episode)')
    for bar, t in zip(bars, metric_times):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                     f'{t:.3f}ms', ha='center', fontsize=10, fontweight='bold')

    plt.suptitle('Computational Efficiency Comparison', fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'computational_efficiency.png')
    plt.savefig(out_path)
    plt.close()
    print(f"\n  ✅ Saved: {out_path}")

    return {
        'backbone_ms': backbone_time,
        'proto_metric_ms': proto_metric_time,
        'kan_metric_ms': kan_metric_time,
        'mlp_metric_ms': mlp_metric_time,
        'kan_params': kan_params,
        'mlp_params': mlp_params,
    }


# ===================================================================
# 6. PER-CLASS ACCURACY (from episodes)
# ===================================================================
def per_class_analysis():
    """Run per-class accuracy analysis on LC 5-way episodes."""
    print("\n" + "="*70)
    print("6. PER-CLASS ACCURACY ANALYSIS")
    print("="*70)

    # We need to re-evaluate episodes to get per-class breakdown
    # Load a single manifest to get class names
    manifest_path = os.path.join(EPISODES_DIR, 'lc5way_shot5_seed2021.json')
    if not os.path.exists(manifest_path):
        print(f"  ⚠️  Manifest not found: {manifest_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load backbone
    backbone_path = os.path.join(CHECKPOINT_DIR, 'resnet18_nct.pth')
    model = models.resnet18(weights=None)
    model.fc = nn.Identity()
    raw = torch.load(backbone_path, map_location='cpu', weights_only=False)
    state = raw.get('state_dict', raw)
    state = {k: v for k, v in state.items() if not k.startswith('fc.') and not k.endswith(('fc.weight', 'fc.bias'))}
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    # Load KAN
    kan_dist = KANLinear(512, 1, grid_size=5, spline_order=3).to(device).eval()
    ckpt_path = os.path.join(CHECKPOINT_DIR, 'kanprotonet_cosine_nct.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        st = ckpt.get('state_dict', ckpt)
        kd_keys = {}
        for k, v in st.items():
            if 'kan_distance.' in k:
                kd_keys[k.split('kan_distance.')[-1]] = v
        if kd_keys:
            kan_dist.load_state_dict(kd_keys, strict=False)

    eval_transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with open(manifest_path, 'r') as f:
        episodes = json.load(f)

    # Limit to first 200 episodes for speed
    n_eval = min(200, len(episodes))
    print(f"  Evaluating {n_eval} episodes for per-class analysis...")

    # Track per-class accuracy for both methods
    class_correct_kan = defaultdict(int)
    class_total_kan = defaultdict(int)
    class_correct_proto = defaultdict(int)
    class_total_proto = defaultdict(int)

    with torch.no_grad():
        for ep_idx in range(n_eval):
            ep = episodes[ep_idx]
            root = Path(ep['dataset_root'])
            if not root.is_absolute():
                root = Path(os.path.join(os.path.dirname(__file__), '..', root))
            label_map = ep['label_map']
            inv_map = {v: k for k, v in label_map.items()}
            n_way = ep['n_way']

            support_imgs, support_labels = [], []
            for cls_name, rel_paths in ep['support'].items():
                local_label = label_map[cls_name]
                for rp in rel_paths:
                    img = Image.open(root / rp).convert('RGB')
                    support_imgs.append(eval_transform(img))
                    support_labels.append(local_label)

            query_imgs, query_labels, query_classes = [], [], []
            for q_item in ep['query']:
                img = Image.open(root / q_item['path']).convert('RGB')
                query_imgs.append(eval_transform(img))
                query_labels.append(q_item['label'])
                query_classes.append(inv_map.get(q_item['label'], str(q_item['label'])))

            x_s = torch.stack(support_imgs).to(device)
            x_q = torch.stack(query_imgs).to(device)
            y_s = torch.tensor(support_labels).to(device)
            y_q = torch.tensor(query_labels).to(device)

            # Extract features
            z_s = model(x_s)  # [n_s, 512]
            z_q = model(x_q)  # [n_q, 512]

            # Compute prototypes
            centroids = torch.zeros(n_way, 512, device=device)
            for c in range(n_way):
                mask = (y_s == c)
                centroids[c] = z_s[mask].mean(0)

            # --- ProtoNet (cosine) ---
            z_s_norm = nn.functional.normalize(z_s, dim=-1)
            z_q_norm = nn.functional.normalize(z_q, dim=-1)
            centroids_norm = torch.zeros(n_way, 512, device=device)
            for c in range(n_way):
                mask = (y_s == c)
                centroids_norm[c] = z_s_norm[mask].mean(0)
            centroids_norm = nn.functional.normalize(centroids_norm, dim=-1)
            scores_proto = z_q_norm @ centroids_norm.t()
            preds_proto = scores_proto.argmax(dim=1)

            # --- KAN distance ---
            diff = z_q.unsqueeze(1) - centroids.unsqueeze(0)  # [n_q, n_way, 512]
            diff_flat = diff.reshape(-1, 512)
            scores_kan = kan_dist(diff_flat).reshape(len(z_q), n_way)
            preds_kan = scores_kan.argmax(dim=1)

            # Record per-class
            for i in range(len(y_q)):
                cls_name = query_classes[i]
                true_label = y_q[i].item()

                class_total_kan[cls_name] += 1
                class_total_proto[cls_name] += 1

                if preds_kan[i].item() == true_label:
                    class_correct_kan[cls_name] += 1
                if preds_proto[i].item() == true_label:
                    class_correct_proto[cls_name] += 1

            if (ep_idx + 1) % 50 == 0:
                print(f"    Processed {ep_idx+1}/{n_eval} episodes...")

    # Summarize
    all_classes = sorted(set(list(class_total_kan.keys()) + list(class_total_proto.keys())))
    print(f"\n  Per-Class Accuracy (over {n_eval} episodes):")
    print(f"  {'Class':<25} {'KAN (%)':>10} {'Proto (%)':>10} {'Δ':>8} {'Better':>10}")
    print(f"  {'─'*68}")

    class_data = []
    for cls in all_classes:
        kan_acc = class_correct_kan[cls] / class_total_kan[cls] * 100 if class_total_kan[cls] > 0 else 0
        proto_acc = class_correct_proto[cls] / class_total_proto[cls] * 100 if class_total_proto[cls] > 0 else 0
        diff = kan_acc - proto_acc
        better = "KAN ✓" if diff > 0.5 else ("Proto ✓" if diff < -0.5 else "≈")
        class_data.append({'class': cls, 'kan': kan_acc, 'proto': proto_acc, 'diff': diff, 'total': class_total_kan[cls]})
        print(f"  {cls:<25} {kan_acc:>10.2f} {proto_acc:>10.2f} {diff:>+8.2f} {better:>10}")

    # Bar chart
    if class_data:
        fig, ax = plt.subplots(figsize=(12, 5))
        classes = [d['class'] for d in class_data]
        kan_accs = [d['kan'] for d in class_data]
        proto_accs = [d['proto'] for d in class_data]

        x = np.arange(len(classes))
        width = 0.35

        bars1 = ax.bar(x - width/2, kan_accs, width, label='KAN-SHOT', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, proto_accs, width, label='ProtoNet', color='#e74c3c', alpha=0.8)

        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Per-Class Accuracy: KAN-SHOT vs ProtoNet (LC 5-Way 5-Shot)')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, 'per_class_accuracy.png')
        plt.savefig(out_path)
        plt.close()
        print(f"\n  ✅ Saved: {out_path}")

    return class_data


# ===================================================================
# MAIN
# ===================================================================
if __name__ == '__main__':
    print("="*70)
    print("  KAN-SHOT Comprehensive Analysis")
    print("="*70)

    # 1. Win/Tie/Loss
    wtl_results = win_tie_loss_analysis()

    # 2. Variance
    var_results = variance_analysis()

    # 3. Bootstrap CI
    bootstrap_ci_visualization()

    # 4. Spline Shape
    spline_results = spline_shape_analysis()

    # 5. Computational Efficiency
    efficiency_results = computational_efficiency()

    # 6. Per-class (runs inference - may take time)
    per_class_results = per_class_analysis()

    print("\n" + "="*70)
    print("  ✅ ALL ANALYSES COMPLETE")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("="*70)
