"""
Evaluation of Frozen Metric Modules (ProtoNet) on Episode Manifests.
ProtoNet has no trainable metric parameters, so it only requires a backbone and a configuration JSON.
"""
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

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kan_shot.methods import ProtoNet


def load_image(img_path: Path, transform):
    img = Image.open(img_path).convert('RGB')
    return transform(img)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Frozen ProtoNet")
    parser.add_argument("--backbone", type=str, required=True, help="Path to resnet18_nct.pth")
    parser.add_argument("--distance_metric", type=str, choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--manifest", type=str, required=True, help="Path to specific episode manifest JSON")
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # 1. Load Backbone
    model = models.resnet18(pretrained=False)
    model.fc = nn.Identity()

    raw_backbone = torch.load(args.backbone, map_location="cpu", weights_only=False)
    state_b = raw_backbone.get("state_dict", raw_backbone)
    state_b = {k: v for k, v in state_b.items() if not k.startswith("fc.") and not k.endswith(("fc.weight", "fc.bias", "classifier.weight", "classifier.bias"))}
    model.load_state_dict(state_b, strict=False)
    model = model.to(device)
    model.eval()

    # 2. Initialize ProtoNet Metric
    m_args = argparse.Namespace(
        distance_metric=args.distance_metric,
        temperature=args.temperature,
        label_smoothing=0.0  # Irrelevant for eval
    )
    
    method = ProtoNet(m_args)
    method = method.to(device)
    method.eval()

    # 3. Transform
    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Load Manifest
    with open(args.manifest, "r") as f:
        episodes = json.load(f)

    print(f"Loaded {len(episodes)} episodes from {args.manifest}")

    accs = []
    with torch.no_grad():
        for ep in tqdm(episodes, desc=f"Evaluating ProtoNet-{args.distance_metric}"):
            root = Path(ep["dataset_root"])
            label_map = ep["label_map"]
            
            # The n_way from manifest overrides config for dynamic evaluation
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

            _, preds_q = method(x_s=x_s, x_q=x_q, y_s=y_s, y_q=y_q, model=model)

            ep_acc = (preds_q == y_q).float().mean().item()
            accs.append(ep_acc)

    accs = np.array(accs)
    mean_acc = np.mean(accs) * 100
    std_acc = np.std(accs) * 100
    ci95 = 1.96 * (std_acc / np.sqrt(len(accs)))

    print(f"\n=======================================================")
    print(f"Results for ProtoNet ({args.distance_metric}, T={args.temperature}) on {args.manifest}:")
    print(f"Mean Accuracy: {mean_acc:.2f}% ± {ci95:.2f}% (95% CI)")
    print(f"=======================================================\n")

    # Save CSV
    import csv
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode_id", "accuracy"])
        for idx, a in enumerate(accs):
            writer.writerow([idx, f"{a:.4f}"])
        writer.writerow(["MEAN", f"{mean_acc:.2f}"])
        writer.writerow(["STD", f"{std_acc:.2f}"])
        writer.writerow(["CI95", f"{ci95:.2f}"])

    print(f"✅ Saved CSV to {args.output}")

if __name__ == "__main__":
    main()
