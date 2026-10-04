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

from kan_shot.methods import KANProtoNet
from src.evaluation.robustness import apply_corruption, CORRUPTION_REGISTRY

def load_image(img_path: Path, transform):
    img = Image.open(img_path).convert('RGB')
    return transform(img)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Robustness for KANProtoNet")
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=84)



    
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.metric, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    m_args = argparse.Namespace(**config)
    
    backbone_path = ckpt.get("backbone_checkpoint", None)
    if backbone_path and "checkpoints" in backbone_path:
        backbone_path = "checkpoints" + backbone_path.split("checkpoints")[-1]
    model = models.resnet18(weights=None)
    model.fc = nn.Identity()
    raw_backbone = torch.load(backbone_path, map_location="cpu", weights_only=False)
    state_b = raw_backbone.get("state_dict", raw_backbone)
    state_b = {k: v for k, v in state_b.items() if not k.startswith("fc.") and not k.endswith(("fc.weight", "fc.bias", "classifier.weight", "classifier.bias"))}
    model.load_state_dict(state_b, strict=False)
    model = model.to(device)
    model.eval()

    method = KANProtoNet(m_args)
    method.load_state_dict(ckpt["method_state_dict"], strict=True)
    method = method.to(device)
    method.eval()

    base_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    with open(args.manifest, "r") as f:
        episodes = json.load(f)

    num_eval = min(args.episodes, len(episodes))
    
    results = {c: {s: [] for s in [1, 2, 3]} for c in CORRUPTION_REGISTRY.keys()}

    print(f"Starting Robustness Evaluation on {num_eval} episodes...")

    with torch.no_grad():
        for i, ep in enumerate(tqdm(episodes[:num_eval], desc="Robustness Episodes")):
            root = Path(ep["dataset_root"])
            if not root.exists() and (Path("..") / root).exists():
                root = Path("..") / root
            label_map = ep["label_map"]
            
            method.n_way = ep["n_way"]

            support_imgs = []
            support_labels = []
            for cls_name, rel_paths in ep["support"].items():
                local_label = label_map[cls_name]
                for rp in rel_paths:
                    img_path = root / rp
                    support_imgs.append(load_image(img_path, base_transform))
                    support_labels.append(local_label)

            query_imgs = []
            query_labels = []
            for q_item in ep["query"]:
                q_path = root / q_item["path"]
                query_imgs.append(load_image(q_path, base_transform))
                query_labels.append(q_item["label"])

            x_s_base = torch.stack(support_imgs).to(device)
            x_q_base = torch.stack(query_imgs).to(device)
            y_s = torch.tensor(support_labels).unsqueeze(0).to(device)
            y_q = torch.tensor(query_labels).unsqueeze(0).to(device)

            for corruption in CORRUPTION_REGISTRY.keys():
                for severity in [1, 2, 3]:
                    x_q_corrupt = apply_corruption(x_q_base, corruption, severity)
                    
                    x_s_norm = norm_transform(x_s_base).unsqueeze(0)
                    x_q_norm = norm_transform(x_q_corrupt).unsqueeze(0)
                    
                    _, preds_q = method(x_s=x_s_norm, x_q=x_q_norm, y_s=y_s, y_q=y_q, model=model)
                    
                    acc = (preds_q == y_q).float().mean().item()
                    results[corruption][severity].append(acc)

    print("\n=======================================================")
    print(f"Robustness Results on {num_eval} episodes ({args.manifest}):")
    
    os.makedirs("results", exist_ok=True)
    with open("results/robustness_results.txt", "w") as f:
        f.write(f"Robustness Evaluation on {args.manifest}\n")
        f.write(f"Episodes: {num_eval}\n\n")
        for corruption in CORRUPTION_REGISTRY.keys():
            print(f"--- {corruption} ---")
            f.write(f"--- {corruption} ---\n")
            for severity in [1, 2, 3]:
                avg_acc = np.mean(results[corruption][severity])
                print(f"  Severity {severity}: {avg_acc*100:.2f}%")
                f.write(f"  Severity {severity}: {avg_acc*100:.2f}%\n")
    print("=======================================================\n")
    print("Saved Robustness summary to results/robustness_results.txt")

if __name__ == "__main__":
    main()
