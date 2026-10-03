"""
Visualize Bottom (Least Important) Spline Curves from KAN Distance Metric.
Generates both bottom-3 spline plots and a side-by-side comparison with top-3.
"""
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kan_shot.methods import KANProtoNet


def main():
    parser = argparse.ArgumentParser(description="Visualize Bottom KAN Splines")
    parser.add_argument("--metric", type=str,
                        default="checkpoints/seed2021/kanprotonet_cosine_nct.pth",
                        help="Path to checkpoint .pth")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.metric}")
    ckpt = torch.load(args.metric, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    m_args = argparse.Namespace(**config)

    # Initialize the metric method
    method = KANProtoNet(m_args)
    method.load_state_dict(ckpt["method_state_dict"], strict=True)
    method.eval()

    if m_args.kan_mode != "distance":
        print("Error: The checkpoint is not trained in 'distance' mode!")
        return

    kan_linear = method.kan_distance

    # Extract weights
    spline_w = kan_linear.spline_weight.detach().numpy()[0]       # [512, num_basis]
    spline_scaler = kan_linear.spline_scaler.detach().numpy()[0]  # [512]
    base_w = np.zeros(512)
    if kan_linear.base_weight is not None:
        base_w = kan_linear.base_weight.detach().numpy()[0]       # [512]

    # Calculate importance scores
    spline_l1 = np.mean(np.abs(spline_w), axis=1)
    importance = np.abs(base_w) + np.abs(spline_scaler) * spline_l1

    # Sort: ascending order for bottom
    sorted_idx = np.argsort(importance)
    bottom_indices = sorted_idx[:3]   # 3 least important
    top_indices = sorted_idx[::-1][:3]  # 3 most important

    print(f"\nTop 3 dimensions:    {top_indices} (scores: {importance[top_indices]})")
    print(f"Bottom 3 dimensions: {bottom_indices} (scores: {importance[bottom_indices]})")
    print(f"Score ratio (top/bottom): {importance[top_indices[0]] / (importance[bottom_indices[0]] + 1e-10):.1f}x\n")

    # Evaluation grid
    grid_points = torch.linspace(-2.0, 2.0, 200).unsqueeze(1)
    x_np = grid_points.squeeze().numpy()

    def silu(x):
        return x * (1 / (1 + np.exp(-x)))

    def compute_spline_curves(dim_idx):
        """Compute base, spline, and total curves for a given dimension."""
        basis_vals = kan_linear.basis(grid_points)
        w_dim = torch.tensor(spline_w[dim_idx]).unsqueeze(0).unsqueeze(0)
        spline_eval = (basis_vals * w_dim).sum(dim=-1).squeeze().numpy()

        scaler = spline_scaler[dim_idx]
        b_w = base_w[dim_idx]

        base_eval = b_w * silu(x_np)
        spline_path = scaler * spline_eval
        total_eval = spline_path + base_eval

        return base_eval, spline_path, total_eval

    # =========================================================================
    # Plot 1: Bottom 3 Splines Only
    # =========================================================================
    print("Generating Bottom 3 Spline Curves...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#7f8c8d', '#95a5a6', '#bdc3c7']  # Gray tones for unimportant dims

    for i in range(3):
        idx = bottom_indices[i]
        base_eval, spline_path, total_eval = compute_spline_curves(idx)

        ax = axes[i]
        ax.plot(x_np, total_eval, color=colors[i], linewidth=3, label="Total Dist Function")
        ax.plot(x_np, base_eval, color='gray', linestyle='--', alpha=0.7, label="Base Linear Path")
        ax.plot(x_np, spline_path, color='lightblue', linestyle='-.', alpha=0.7, label="Spline Path")
        ax.set_title(f"Dim {idx} (Rank {512 - i}/512)", fontweight='bold', fontsize=12)
        ax.set_xlabel("Feature Difference (Query - Prototype)")
        ax.set_ylabel("Distance Penalty Score")
        ax.axvline(0, color='black', linewidth=1, alpha=0.3)
        ax.axhline(0, color='black', linewidth=1, alpha=0.3)

        # Show the importance score
        ax.text(0.05, 0.95, f"SFA Score: {importance[idx]:.4f}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Bottom 3 Least Important Dimensions — KAN Spline Curves",
                 fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    path1 = os.path.join(args.output_dir, "kan_bottom_splines.png")
    plt.savefig(path1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved Bottom Splines plot to {path1}")

    # =========================================================================
    # Plot 2: Side-by-Side Comparison (Top 3 vs Bottom 3)
    # =========================================================================
    print("Generating Top vs Bottom Comparison...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    top_colors = ['#a60845', '#265940', '#d16d11']
    bottom_colors = ['#7f8c8d', '#95a5a6', '#bdc3c7']

    # Row 1: Top 3
    for i in range(3):
        idx = top_indices[i]
        base_eval, spline_path, total_eval = compute_spline_curves(idx)

        ax = axes[0, i]
        ax.plot(x_np, total_eval, color=top_colors[i], linewidth=3, label="Total")
        ax.plot(x_np, base_eval, color='gray', linestyle='--', alpha=0.7, label="Base")
        ax.plot(x_np, spline_path, color='lightblue', linestyle='-.', alpha=0.7, label="Spline")
        ax.set_title(f"TOP — Dim {idx} (Rank {i+1})", fontweight='bold', fontsize=12,
                     color=top_colors[i])
        ax.set_xlabel("Feature Difference (δ)")
        ax.set_ylabel("Distance Penalty")
        ax.axvline(0, color='black', linewidth=1, alpha=0.3)
        ax.axhline(0, color='black', linewidth=1, alpha=0.3)
        ax.text(0.05, 0.95, f"SFA: {importance[idx]:.4f}",
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
        if i == 0:
            ax.legend(fontsize=8)

    # Row 2: Bottom 3
    for i in range(3):
        idx = bottom_indices[i]
        base_eval, spline_path, total_eval = compute_spline_curves(idx)

        ax = axes[1, i]
        ax.plot(x_np, total_eval, color=bottom_colors[i], linewidth=3, label="Total")
        ax.plot(x_np, base_eval, color='gray', linestyle='--', alpha=0.7, label="Base")
        ax.plot(x_np, spline_path, color='lightblue', linestyle='-.', alpha=0.7, label="Spline")
        ax.set_title(f"BOTTOM — Dim {idx} (Rank {512-i})", fontweight='bold', fontsize=12,
                     color=bottom_colors[i])
        ax.set_xlabel("Feature Difference (δ)")
        ax.set_ylabel("Distance Penalty")
        ax.axvline(0, color='black', linewidth=1, alpha=0.3)
        ax.axhline(0, color='black', linewidth=1, alpha=0.3)
        ax.text(0.05, 0.95, f"SFA: {importance[idx]:.4f}",
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#ccccff', alpha=0.8))
        if i == 0:
            ax.legend(fontsize=8)

    # Match Y-axis scale across all subplots for fair comparison
    all_ylims = []
    for row in axes:
        for ax in row:
            all_ylims.extend(ax.get_ylim())
    global_ymin, global_ymax = min(all_ylims), max(all_ylims)
    for row in axes:
        for ax in row:
            ax.set_ylim(global_ymin, global_ymax)

    fig.suptitle("KAN Spline Comparison: Top 3 (Important) vs Bottom 3 (Unimportant)",
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    path2 = os.path.join(args.output_dir, "kan_top_vs_bottom_splines.png")
    plt.savefig(path2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved Top vs Bottom comparison to {path2}")

    # =========================================================================
    # Print summary statistics
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Dimension':<12} {'Rank':<8} {'SFA Score':<12} {'|base_w|':<12} {'|scaler|':<12}")
    print("-"*60)
    for i, idx in enumerate(top_indices):
        print(f"Dim {idx:<6}   #{i+1:<5}   {importance[idx]:<12.6f} {abs(base_w[idx]):<12.6f} {abs(spline_scaler[idx]):<12.6f}")
    print("-"*60)
    for i, idx in enumerate(bottom_indices):
        print(f"Dim {idx:<6}   #{512-i:<5}   {importance[idx]:<12.6f} {abs(base_w[idx]):<12.6f} {abs(spline_scaler[idx]):<12.6f}")
    print("="*60)


if __name__ == "__main__":
    main()
