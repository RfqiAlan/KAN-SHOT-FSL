"""
LayerCAM Visualization: ProtoNet vs KAN-SHOT
Generates LayerCAM heatmaps for qualitative comparison across:
  - Near-Domain  (CRC-VAL, 9-Way 5-Shot)
  - Cross-Domain / Domain Shift (LC25000, 5-Way 5-Shot)

Usage:
  python scripts/visualize_layercam.py \
      --backbone checkpoints/seed2021/resnet18_nct.pth \
      --kan_metric checkpoints/seed2021/kanprotonet_cosine_nct.pth \
      --near_manifest episodes/crc_val_shot5_seed2021.json \
      --cross_manifest episodes/lc5way_shot5_seed2021.json \
      --output_dir results/layercam
"""
import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kan_shot.methods import KANProtoNet
from kan_shot.methods.utils import extract_features, compute_centroids

# ---------------------------------------------------------------------------
# pytorch-grad-cam imports
# ---------------------------------------------------------------------------
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================================
# Custom wrapper: makes ResNet backbone compatible with pytorch-grad-cam
# by outputting *negative distance to a target prototype* as the "logit".
# ============================================================================
class ProtoNetCAMWrapper(nn.Module):
    """
    Wraps the ResNet backbone so that its forward() returns a score vector
    (negative distance to each prototype).  pytorch-grad-cam will backprop
    through this to obtain spatial gradients.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone          # ResNet-18 (fc = Identity)
        self.prototypes = None            # set externally per episode
        self.distance_metric = "cosine"

    def set_prototypes(self, prototypes: torch.Tensor, distance_metric="cosine"):
        """prototypes: [n_way, feat_dim]"""
        self.prototypes = prototypes
        self.distance_metric = distance_metric

    def forward(self, x):
        """x: [B, 3, H, W]  -> scores: [B, n_way]"""
        z = self.backbone(x)  # [B, feat_dim]

        if self.distance_metric == "cosine":
            z_norm = F.normalize(z, p=2, dim=-1)
            p_norm = F.normalize(self.prototypes, p=2, dim=-1)
            scores = z_norm @ p_norm.T  # [B, n_way]
        else:  # l2
            scores = -((z.unsqueeze(1) - self.prototypes.unsqueeze(0)) ** 2).sum(-1)

        return scores


class KANProtoNetCAMWrapper(nn.Module):
    """
    Same idea but routes the backbone features through the KAN distance head
    so that the heatmap reflects the KAN-learned metric.
    """

    def __init__(self, backbone: nn.Module, kan_method: KANProtoNet):
        super().__init__()
        self.backbone = backbone
        self.kan_method = kan_method
        self.prototypes_raw = None  # raw support centroids [n_way, d]

    def set_prototypes(self, z_support: torch.Tensor, y_support: torch.Tensor, n_way: int):
        """Compute and store prototypes from support features."""
        # z_support: [1, n_support, d],  y_support: [1, n_support]
        if self.kan_method.kan_prenorm:
            z_support = F.normalize(z_support, p=2, dim=-1)

        if self.kan_method.kan_mode == 'distance':
            centroids = compute_centroids(z_support, y_support, n_way=n_way)
            self.prototypes_raw = centroids.squeeze(0)  # [n_way, d]
        elif self.kan_method.kan_mode in ('transform', 'dual'):
            z_support_kan = self.kan_method.kan_transform(z_support)
            if self.kan_method.kan_mode == 'dual':
                centroids = compute_centroids(z_support_kan, y_support, n_way=n_way)
                self.prototypes_raw = centroids.squeeze(0)
            else:
                centroids = compute_centroids(z_support_kan, y_support, n_way=n_way)
                centroids_norm = F.normalize(centroids, p=2, dim=-1)
                self.prototypes_raw = centroids_norm.squeeze(0)

    def forward(self, x):
        """x: [B, 3, H, W] -> scores: [B, n_way]"""
        z = self.backbone(x)  # [B, d]

        if self.kan_method.kan_prenorm:
            z = F.normalize(z, p=2, dim=-1)

        if self.kan_method.kan_mode == 'distance':
            # diff -> kan_distance
            diff = z.unsqueeze(1) - self.prototypes_raw.unsqueeze(0)  # [B, n_way, d]
            B, N, D = diff.shape
            scores_flat = self.kan_method.kan_distance(diff.reshape(B * N, D))
            scores = scores_flat.view(B, N) / self.kan_method.temperature
        elif self.kan_method.kan_mode in ('transform', 'dual'):
            z_kan = self.kan_method.kan_transform(z.unsqueeze(0)).squeeze(0)
            if self.kan_method.kan_mode == 'dual':
                diff = z_kan.unsqueeze(1) - self.prototypes_raw.unsqueeze(0)
                B, N, D = diff.shape
                scores_flat = self.kan_method.kan_distance(diff.reshape(B * N, D))
                scores = scores_flat.view(B, N) / self.kan_method.temperature
            else:
                z_kan_norm = F.normalize(z_kan, p=2, dim=-1)
                scores = z_kan_norm @ self.prototypes_raw.T
        else:
            raise ValueError(f"Unknown kan_mode: {self.kan_method.kan_mode}")

        return scores


# ============================================================================
# Target class for pytorch-grad-cam: selects the predicted class score
# ============================================================================
class PredictedClassTarget:
    """Return the score of the predicted (argmax) class.
    pytorch-grad-cam calls this per-sample, so model_output is 1D [n_way].
    """
    def __call__(self, model_output):
        # model_output: [n_way] (1D per sample)
        return model_output[model_output.argmax()]


# ============================================================================
# Helpers
# ============================================================================
def load_image_pil(img_path, size=84):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    return img


def pil_to_tensor(img, transform):
    return transform(img)


def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Convert normalized tensor back to [0, 1] numpy for overlay."""
    t = tensor.clone()
    for ch, m, s in zip(t, mean, std):
        ch.mul_(s).add_(m)
    return t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def build_backbone(backbone_path, device):
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
    return model


def build_kan_method(metric_path, device):
    ckpt = torch.load(metric_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    m_args = argparse.Namespace(**config)
    method = KANProtoNet(m_args)
    method.load_state_dict(ckpt["method_state_dict"], strict=True)
    method = method.to(device)
    method.eval()
    return method, config


def load_episode_data(episode, transform, device):
    """Load a single episode's support & query images/labels."""
    root = Path(episode["dataset_root"])
    label_map = episode["label_map"]
    n_way = episode["n_way"]

    support_imgs, support_labels = [], []
    for cls_name, rel_paths in episode["support"].items():
        local_label = label_map[cls_name]
        for rp in rel_paths:
            img_path = root / rp
            support_imgs.append(pil_to_tensor(load_image_pil(img_path), transform))
            support_labels.append(local_label)

    query_imgs_pil = []
    query_imgs_tensor = []
    query_labels = []
    query_class_names = []
    for q_item in episode["query"]:
        q_path = root / q_item["path"]
        pil_img = load_image_pil(q_path)
        query_imgs_pil.append(pil_img)
        query_imgs_tensor.append(pil_to_tensor(pil_img, transform))
        query_labels.append(q_item["label"])
        # find class name from label
        inv_map = {v: k for k, v in label_map.items()}
        query_class_names.append(inv_map[q_item["label"]])

    x_s = torch.stack(support_imgs).unsqueeze(0).to(device)
    x_q = torch.stack(query_imgs_tensor).to(device)  # [n_query, 3, H, W]
    y_s = torch.tensor(support_labels).unsqueeze(0).to(device)

    return x_s, x_q, y_s, query_imgs_pil, query_labels, query_class_names, n_way, label_map


def generate_layercam_grid(
    backbone, kan_method,
    episode, transform, device,
    domain_label, output_dir,
    num_queries=5, episode_idx=0
):
    """
    For ONE episode, generate a comparison grid:
       Row 0: Original images
       Row 1: ProtoNet LayerCAM
       Row 2: KAN-SHOT LayerCAM
    Saves the figure to output_dir.
    """
    x_s, x_q, y_s, q_pils, q_labels, q_cls_names, n_way, label_map = \
        load_episode_data(episode, transform, device)

    inv_map = {v: k for k, v in label_map.items()}

    # ---- Pick a diverse subset of query images (one per class if possible) ----
    selected_indices = []
    seen_classes = set()
    for idx, lbl in enumerate(q_labels):
        if lbl not in seen_classes and len(selected_indices) < num_queries:
            selected_indices.append(idx)
            seen_classes.add(lbl)
    # fill remaining slots
    for idx in range(len(q_labels)):
        if idx not in selected_indices and len(selected_indices) < num_queries:
            selected_indices.append(idx)

    selected_indices = selected_indices[:num_queries]

    # ---- Compute support features & prototypes for ProtoNet ----
    with torch.no_grad():
        z_s = extract_features(x_s, backbone)  # [1, n_support, d]

    # ProtoNet wrapper
    proto_wrapper = ProtoNetCAMWrapper(backbone).to(device)
    proto_wrapper.eval()
    z_s_norm = F.normalize(z_s, p=2, dim=-1)
    protos = compute_centroids(z_s_norm, y_s, n_way=n_way)
    protos_norm = F.normalize(protos, p=2, dim=-1)
    proto_wrapper.set_prototypes(protos_norm.squeeze(0), "cosine")

    # KAN wrapper
    kan_wrapper = KANProtoNetCAMWrapper(backbone, kan_method).to(device)
    kan_wrapper.eval()
    kan_wrapper.set_prototypes(z_s, y_s, n_way)

    # ---- Target layers for LayerCAM ----
    # Use multiple layers for richer heatmaps (layer1 through layer4)
    target_layers_proto = [
        proto_wrapper.backbone.layer1[-1],
        proto_wrapper.backbone.layer2[-1],
        proto_wrapper.backbone.layer3[-1],
        proto_wrapper.backbone.layer4[-1],
    ]
    target_layers_kan = [
        kan_wrapper.backbone.layer1[-1],
        kan_wrapper.backbone.layer2[-1],
        kan_wrapper.backbone.layer3[-1],
        kan_wrapper.backbone.layer4[-1],
    ]

    cam_target = [PredictedClassTarget()]

    # ---- Generate heatmaps ----
    proto_cams = []
    kan_cams = []

    with LayerCAM(model=proto_wrapper, target_layers=target_layers_proto) as cam_proto:
        for idx in selected_indices:
            inp = x_q[idx].unsqueeze(0)  # [1, 3, H, W]
            grayscale_cam = cam_proto(input_tensor=inp, targets=cam_target)
            proto_cams.append(grayscale_cam[0])  # [H, W]

    with LayerCAM(model=kan_wrapper, target_layers=target_layers_kan) as cam_kan:
        for idx in selected_indices:
            inp = x_q[idx].unsqueeze(0)
            grayscale_cam = cam_kan(input_tensor=inp, targets=cam_target)
            kan_cams.append(grayscale_cam[0])

    # ---- Build the comparison figure ----
    n = len(selected_indices)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))

    if n == 1:
        axes = axes.reshape(3, 1)

    row_labels = ["Original", "ProtoNet\nLayerCAM", "KAN-SHOT\nLayerCAM"]

    for col, idx in enumerate(selected_indices):
        # Original image
        rgb_img = denormalize(x_q[idx].cpu())
        axes[0, col].imshow(rgb_img)
        cls_name = q_cls_names[idx].replace("_", " ").title()
        axes[0, col].set_title(cls_name, fontsize=12, fontweight="bold")

        # ProtoNet heatmap
        overlay_proto = show_cam_on_image(rgb_img, proto_cams[col], use_rgb=True)
        axes[1, col].imshow(overlay_proto)

        # KAN-SHOT heatmap
        overlay_kan = show_cam_on_image(rgb_img, kan_cams[col], use_rgb=True)
        axes[2, col].imshow(overlay_kan)

    for row in range(3):
        for col in range(n):
            axes[row, col].axis("off")

    fig.suptitle(f"LayerCAM Comparison \u2014 {domain_label}", fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0.07, 0.0, 1.0, 0.94])

    # Row labels placed right next to images using fig.text()
    for row, label in enumerate(row_labels):
        bbox = axes[row, 0].get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        x_pos = bbox.x0 - 0.01  # small gap
        fig.text(x_pos, y_center, label, fontsize=14, fontweight="bold",
                 ha="center", va="center", rotation=90)

    out_path = os.path.join(output_dir, f"layercam_{domain_label.lower().replace(' ', '_').replace('-', '_')}_ep{episode_idx}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="LayerCAM Visualization: ProtoNet vs KAN-SHOT")
    parser.add_argument("--backbone", type=str, default="checkpoints/seed2021/resnet18_nct.pth")
    parser.add_argument("--kan_metric", type=str, default="checkpoints/seed2021/kanprotonet_cosine_nct.pth")
    parser.add_argument("--near_manifest", type=str, default="episodes/crc_val_shot5_seed2021.json",
                        help="Near-domain episode manifest (CRC-VAL)")
    parser.add_argument("--cross_manifest", type=str, default="episodes/lc5way_shot5_seed2021.json",
                        help="Cross-domain episode manifest (LC25000)")
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--num_queries", type=int, default=5,
                        help="Number of query images to show per domain")
    parser.add_argument("--episode_idx", type=int, default=0,
                        help="Which episode index to visualize")
    parser.add_argument("--output_dir", type=str, default="results/layercam")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Build models ----
    print("Loading backbone...")
    backbone = build_backbone(args.backbone, device)

    print("Loading KAN-SHOT metric...")
    kan_method, kan_config = build_kan_method(args.kan_metric, device)

    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ---- Near-Domain (CRC-VAL) ----
    print("\n=== Near-Domain: CRC-VAL ===")
    with open(args.near_manifest, "r") as f:
        near_episodes = json.load(f)
    near_ep = near_episodes[args.episode_idx]

    generate_layercam_grid(
        backbone=backbone,
        kan_method=kan_method,
        episode=near_ep,
        transform=eval_transform,
        device=device,
        domain_label="Near-Domain (CRC-VAL 9-Way 5-Shot)",
        output_dir=args.output_dir,
        num_queries=args.num_queries,
        episode_idx=args.episode_idx,
    )

    # ---- Cross-Domain / Domain Shift (LC25000) ----
    print("\n=== Cross-Domain: LC25000 (Domain Shift) ===")
    with open(args.cross_manifest, "r") as f:
        cross_episodes = json.load(f)
    cross_ep = cross_episodes[args.episode_idx]

    generate_layercam_grid(
        backbone=backbone,
        kan_method=kan_method,
        episode=cross_ep,
        transform=eval_transform,
        device=device,
        domain_label="Cross-Domain (LC 5-Way 5-Shot)",
        output_dir=args.output_dir,
        num_queries=args.num_queries,
        episode_idx=args.episode_idx,
    )

    print(f"\nAll figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
