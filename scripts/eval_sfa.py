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
from kan_shot.methods.utils import extract_features
from src.evaluation.sfa import SplineFeatureAttribution

def load_image(img_path: Path, transform):
    img = Image.open(img_path).convert('RGB')
    return transform(img)

def main():
    parser = argparse.ArgumentParser(description="Evaluate SFA for KANProtoNet")
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Checkpoints
    ckpt = torch.load(args.metric, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    m_args = argparse.Namespace(**config)
    
    backbone_path = ckpt.get("backbone_checkpoint", None)
    model = models.resnet18(pretrained=False)
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

    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with open(args.manifest, "r") as f:
        episodes = json.load(f)

    num_eval = min(args.episodes, len(episodes))
    
    sfa = SplineFeatureAttribution(method, num_steps=20)
    
    all_ins_auc = []
    all_del_auc = []
    all_importances = []

    print(f"Starting SFA Evaluation on {num_eval} episodes...")

    with torch.no_grad():
        for i, ep in enumerate(tqdm(episodes[:num_eval], desc="SFA Episodes")):
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

            z_s = extract_features(x_s, model)
            z_q = extract_features(x_q, model)

            res = sfa.full_evaluation(z_q, z_s, y_s, y_q, model)
            all_ins_auc.append(res.insertion_auc)
            all_del_auc.append(res.deletion_auc)
            all_importances.append(res.feature_importance.cpu().numpy())

    avg_ins = np.mean(all_ins_auc)
    avg_del = np.mean(all_del_auc)
    
    print("\n=======================================================")
    print(f"SFA Results on {num_eval} episodes ({args.manifest}):")
    print(f"Mean Insertion AUC: {avg_ins:.4f} (Higher is better, meaning top KAN features carry most signal)")
    print(f"Mean Deletion AUC:  {avg_del:.4f} (Lower is better, meaning removing KAN features breaks accuracy)")
    print("=======================================================\n")
    
    os.makedirs("results", exist_ok=True)
    with open("results/sfa_results.txt", "w") as f:
        f.write(f"SFA Evaluation on {args.manifest}\n")
        f.write(f"Episodes: {num_eval}\n")
        f.write(f"Mean Insertion AUC: {avg_ins:.4f}\n")
        f.write(f"Mean Deletion AUC: {avg_del:.4f}\n")
        
    print("✅ Saved SFA summary to results/sfa_results.txt")

if __name__ == "__main__":
    main()
