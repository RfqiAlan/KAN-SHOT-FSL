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
    parser = argparse.ArgumentParser(description="Visualize KAN Distance Interpretability")
    parser.add_argument("--metric", type=str, default="checkpoints/seed2021/kanprotonet_cosine_nct.pth", help="Path to checkpoint .pth")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save plots")
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

    # Verify we are in distance mode
    if m_args.kan_mode != "distance":
        print("Error: The checkpoint is not trained in 'distance' mode!")
        return

    kan_linear = method.kan_distance
    
    print("Extracting weights from KANLinear...")
    # Shapes:
    # spline_weight: [1, 512, num_basis]
    # base_weight: [1, 512]
    # spline_scaler: [1, 512]
    
    spline_w = kan_linear.spline_weight.detach().numpy()[0]  # [512, num_basis]
    spline_scaler = kan_linear.spline_scaler.detach().numpy()[0] # [512]
    
    base_w = np.zeros(512)
    if kan_linear.base_weight is not None:
        base_w = kan_linear.base_weight.detach().numpy()[0] # [512]

    # Calculate Importance Score for each of the 512 dimensions
    # Importance = |base_weight| + |spline_scaler| * L1_Norm(spline_weight)
    spline_l1 = np.mean(np.abs(spline_w), axis=1) # Average magnitude of spline coefficients for each dim
    importance = np.abs(base_w) + np.abs(spline_scaler) * spline_l1
    
    # Sort features by importance
    sorted_idx = np.argsort(importance)[::-1]
    top_k = 20
    top_indices = sorted_idx[:top_k]
    top_scores = importance[top_indices]

    # 1. Bar Chart of Feature Importance
    print("Generating Feature Importance Bar Chart...")
    plt.figure(figsize=(12, 6))
    bars = plt.bar(np.arange(top_k), top_scores, color='#5fbdbb')
    plt.xticks(np.arange(top_k), [f"Dim {idx}" for idx in top_indices], rotation=45, ha='right')
    plt.title("Top 20 Most Important Features in KAN-Distance Metric", fontsize=14, fontweight='bold')
    plt.ylabel("Importance Score (Weight Magnitude)", fontsize=12)
    plt.xlabel("ResNet18 Feature Dimension", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    importance_plot_path = os.path.join(args.output_dir, "kan_feature_importance.png")
    plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved Feature Importance plot to {importance_plot_path}")

    # 2. Plot 1D Splines for Top 3 Features
    print("Generating 1D Spline Curves for Top Features...")
    # Evaluate splines over [-2, 2] since inputs are differences between L2-normalized vectors
    grid_points = torch.linspace(-2.0, 2.0, 100).unsqueeze(1) # [100, 1]
    
    plt.figure(figsize=(15, 5))
    colors = ['#a60845', '#265940', '#d16d11']
    
    for i in range(min(3, top_k)):
        idx = top_indices[i]
        
        # Calculate basis functions for this grid
        # basis is a BSplineBasis module inside KANLinear
        basis_vals = kan_linear.basis(grid_points) # [100, 1, num_basis]
        
        # Multiply by weights for this specific dimension
        w_dim = torch.tensor(spline_w[idx]).unsqueeze(0).unsqueeze(0) # [1, 1, num_basis]
        
        spline_eval = (basis_vals * w_dim).sum(dim=-1).squeeze().numpy() # [100]
        
        # Scale and add base function
        scaler = spline_scaler[idx]
        b_w = base_w[idx]
        
        # silu activation for base path
        x_np = grid_points.squeeze().numpy()
        def silu(x): return x * (1 / (1 + np.exp(-x)))
        base_eval = b_w * silu(x_np)
        
        total_eval = (scaler * spline_eval) + base_eval
        
        plt.subplot(1, 3, i+1)
        plt.plot(x_np, total_eval, color=colors[i], linewidth=3, label=f"Total Dist Function")
        plt.plot(x_np, base_eval, color='gray', linestyle='--', label="Base Linear Path")
        plt.plot(x_np, scaler * spline_eval, color='lightblue', linestyle='-.', label="Spline Path")
        plt.title(f"Dim {idx} (Rank {i+1})", fontweight='bold')
        plt.xlabel("Feature Difference (Query - Prototype)")
        plt.ylabel("Distance Penalty Score")
        plt.axvline(0, color='black', linewidth=1, alpha=0.3)
        plt.axhline(0, color='black', linewidth=1, alpha=0.3)
        if i == 0:
            plt.legend()
            
    plt.suptitle("How KAN Penalizes Distance Differences in Top Dimensions", fontsize=16, y=1.05)
    plt.tight_layout()
    
    splines_plot_path = os.path.join(args.output_dir, "kan_top_splines.png")
    plt.savefig(splines_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved Top Splines plot to {splines_plot_path}")

if __name__ == "__main__":
    main()
