import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kan_shot.methods import KANProtoNet

def load_image(img_path: Path, transform):
    img = Image.open(img_path).convert('RGB')
    return transform(img)

def main():
    parser = argparse.ArgumentParser(description="Generate t-SNE and Confusion Matrix")
    parser.add_argument("--metric", type=str, default="checkpoints/seed2021/kanprotonet_cosine_nct.pth")
    parser.add_argument("--manifest", type=str, default="episodes/lc5way_shot5_seed2021.json")
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to process for CM")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Checkpoint
    ckpt = torch.load(args.metric, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    m_args = argparse.Namespace(**config)
    
    # 2. Load Backbone
    backbone_path = ckpt.get("backbone_checkpoint", None)
    model = models.resnet18(pretrained=False)
    model.fc = nn.Identity()
    raw_backbone = torch.load(backbone_path, map_location="cpu", weights_only=False)
    state_b = raw_backbone.get("state_dict", raw_backbone)
    state_b = {k: v for k, v in state_b.items() if not k.startswith("fc.") and not k.endswith(("fc.weight", "fc.bias", "classifier.weight", "classifier.bias"))}
    model.load_state_dict(state_b, strict=False)
    model = model.to(device)
    model.eval()

    # 3. Initialize Metric Method
    method = KANProtoNet(m_args)
    method.load_state_dict(ckpt["method_state_dict"], strict=True)
    method = method.to(device)
    method.eval()

    # 4. Transform
    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 5. Load Manifest
    with open(args.manifest, "r") as f:
        episodes = json.load(f)

    # 6. Evaluate and Collect Data
    all_preds = []
    all_trues = []
    
    # Extract class names from the first episode to label the CM
    first_ep = episodes[0]
    class_names = list(first_ep["support"].keys()) # original class names like "lung_aca", "lung_n"
    label_map_inv = {v: k for k, v in first_ep["label_map"].items()}
    ordered_class_names = [label_map_inv[i] for i in range(len(label_map_inv))]

    tsne_features = None
    tsne_labels = None
    tsne_centroids = None

    num_eval = min(args.episodes, len(episodes))
    with torch.no_grad():
        for i, ep in enumerate(tqdm(episodes[:num_eval], desc="Evaluating Episodes")):
            root = Path(ep["dataset_root"])
            label_map = ep["label_map"]
            
            m_args.n_way = ep["n_way"]
            method.n_way = ep["n_way"]

            support_imgs = []
            support_labels = []
            for cls_name, rel_paths in ep["support"].items():
                local_label = label_map[cls_name]
                for rp in rel_paths:
                    img_path = root / rp
                    support_imgs.append(load_image(img_path, eval_transform))
                    support_labels.append(local_label)

            query_imgs = []
            query_labels = []
            for q_item in ep["query"]:
                q_path = root / q_item["path"]
                query_imgs.append(load_image(q_path, eval_transform))
                query_labels.append(q_item["label"])

            x_s = torch.stack(support_imgs).unsqueeze(0).to(device)
            x_q = torch.stack(query_imgs).unsqueeze(0).to(device)
            y_s = torch.tensor(support_labels).unsqueeze(0).to(device)
            y_q = torch.tensor(query_labels).unsqueeze(0).to(device)

            logits, preds_q = method(x_s=x_s, x_q=x_q, y_s=y_s, y_q=y_q, model=model)

            all_preds.extend(preds_q.squeeze(0).cpu().numpy())
            all_trues.extend(y_q.squeeze(0).cpu().numpy())

            # Save features for t-SNE only from the first episode
            if i == 0:
                from kan_shot.methods.utils import extract_features, compute_centroids
                z_s = extract_features(x_s, model)
                z_q = extract_features(x_q, model)
                if method.kan_prenorm:
                    import torch.nn.functional as F
                    z_s = F.normalize(z_s, p=2, dim=-1)
                    z_q = F.normalize(z_q, p=2, dim=-1)
                
                # Compute centroids
                centroids = compute_centroids(z_s, y_s, n_way=method.n_way)
                
                tsne_features = z_q.squeeze(0).cpu().numpy()
                tsne_labels = y_q.squeeze(0).cpu().numpy()
                tsne_centroids = centroids.squeeze(0).cpu().numpy()

    # --- Generate Confusion Matrix ---
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(all_trues, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=ordered_class_names, yticklabels=ordered_class_names)
    plt.title("Confusion Matrix: LC 5-Way 5-Shot (KAN-Distance)", fontsize=16, fontweight='bold')
    plt.ylabel("True Class", fontsize=14)
    plt.xlabel("Predicted Class", fontsize=14)
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, "kan_lc_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\u2705 Saved Confusion Matrix to {cm_path}")

    # --- Generate t-SNE ---
    print("Generating t-SNE for Episode 0...")
    # Combine features and centroids for t-SNE
    all_points = np.vstack([tsne_features, tsne_centroids])
    num_queries = tsne_features.shape[0]
    
    tsne = TSNE(n_components=2, perplexity=15, random_state=42)
    points_2d = tsne.fit_transform(all_points)
    
    features_2d = points_2d[:num_queries]
    centroids_2d = points_2d[num_queries:]
    
    plt.figure(figsize=(12, 10))
    palette = sns.color_palette("husl", method.n_way)
    
    for c_idx in range(method.n_way):
        # Plot queries
        idx = (tsne_labels == c_idx)
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], 
                    label=f"Query: {ordered_class_names[c_idx]}", 
                    color=palette[c_idx], alpha=0.6, s=50)
        # Plot centroids
        plt.scatter(centroids_2d[c_idx, 0], centroids_2d[c_idx, 1], 
                    color=palette[c_idx], marker='X', s=300, edgecolors='black', linewidth=1.5,
                    label=f"Centroid: {ordered_class_names[c_idx]}")
                    
    plt.title("t-SNE Feature Embeddings & Centroids (Episode 0)", fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    tsne_path = os.path.join(args.output_dir, "kan_lc_tsne.png")
    plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\u2705 Saved t-SNE to {tsne_path}")

if __name__ == "__main__":
    main()
