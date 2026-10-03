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
import copy
import scipy.stats

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kan_shot.methods import KANProtoNet

def load_image(img_path: Path, transform):
    img = Image.open(img_path).convert('RGB')
    return transform(img)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def main():
    parser = argparse.ArgumentParser(description="Evaluate KAN-SHOT with Episodic Fine-Tuning")
    parser.add_argument("--metric", type=str, default="checkpoints/seed2021/kanprotonet_cosine_nct.pth")
    parser.add_argument("--manifest", type=str, default="episodes/lc5way_shot5_seed2021.json")
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to evaluate")
    parser.add_argument("--ft_steps", type=int, default=50, help="Number of fine-tuning steps per episode")
    parser.add_argument("--ft_lr", type=float, default=0.0005, help="Learning rate for fine-tuning")
    parser.add_argument("--ft_layers", type=str, default="layer4", choices=["kan", "layer4", "all", "all_freeze_spline"], help="Which parts of the model to fine-tune")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Base Checkpoints
    ckpt = torch.load(args.metric, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    m_args = argparse.Namespace(**config)
    
    # Load Backbone
    backbone_path = ckpt.get("backbone_checkpoint", None)
    base_model = models.resnet18(pretrained=False)
    base_model.fc = nn.Identity()
    raw_backbone = torch.load(backbone_path, map_location="cpu", weights_only=False)
    state_b = raw_backbone.get("state_dict", raw_backbone)
    state_b = {k: v for k, v in state_b.items() if not k.startswith("fc.") and not k.endswith(("fc.weight", "fc.bias", "classifier.weight", "classifier.bias"))}
    base_model.load_state_dict(state_b, strict=False)
    base_model = base_model.to(device)

    # Initialize Base Metric Method
    base_method = KANProtoNet(m_args)
    base_method.load_state_dict(ckpt["method_state_dict"], strict=True)
    base_method = base_method.to(device)

    # Transform
    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Manifest
    with open(args.manifest, "r") as f:
        episodes = json.load(f)

    num_eval = min(args.episodes, len(episodes))
    all_acc = []

    print(f"Starting Episodic Fine-Tuning on {num_eval} episodes...")
    print(f"FT Steps: {args.ft_steps}, FT LR: {args.ft_lr}")

    for i, ep in enumerate(tqdm(episodes[:num_eval], desc="Evaluating Episodes")):
        root = Path(ep["dataset_root"])
        label_map = ep["label_map"]
        
        m_args.n_way = ep["n_way"]
        
        # Load Images
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

        # ---------------------------------------------------------
        # EPISODIC FINE-TUNING
        # ---------------------------------------------------------
        # Clone base models to avoid polluting weights across episodes
        model = copy.deepcopy(base_model)
        method = copy.deepcopy(base_method)
        method.n_way = ep["n_way"]

        # Freeze everything first
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze based on ft_layers
        if args.ft_layers in ["all", "all_freeze_spline"]:
            for param in model.parameters():
                param.requires_grad = True
        elif args.ft_layers == "layer4":
            for param in model.layer4.parameters():
                param.requires_grad = True
        # if "kan", model remains completely frozen
            
        for name, param in method.named_parameters():
            if args.ft_layers == "all_freeze_spline" and "spline_weight" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        trainable_params = [p for p in model.parameters() if p.requires_grad] + \
                           [p for p in method.parameters() if p.requires_grad]
        
        optimizer = torch.optim.Adam(trainable_params, lr=args.ft_lr)

        model.train()
        method.train()
        
        # Micro-training loop on support set
        for step in range(args.ft_steps):
            optimizer.zero_grad()
            # We use x_s as both support and query for calculating prototypical loss
            # This forces the features to form tight clusters around their centroids
            loss, _ = method(x_s=x_s, x_q=x_s, y_s=y_s, y_q=y_s, model=model)
            loss.backward()
            optimizer.step()

        # ---------------------------------------------------------
        # EVALUATION
        # ---------------------------------------------------------
        model.eval()
        method.eval()
        with torch.no_grad():
            _, preds_q = method(x_s=x_s, x_q=x_q, y_s=y_s, y_q=y_q, model=model)
            
        acc = (preds_q == y_q).float().mean().item()
        all_acc.append(acc)

    # Print Results
    mean_acc, conf_int = mean_confidence_interval(all_acc)
    print("\n=======================================================")
    print(f"Results for Fine-Tuned Model on {num_eval} episodes:")
    print(f"Mean Accuracy: {mean_acc * 100:.2f}% ± {conf_int * 100:.2f}% (95% CI)")
    print("=======================================================\n")

    # Save CSV
    out_file = os.path.join(args.output_dir, f"kan_lc5way_5shot_finetune_{args.ft_layers}.csv")
    with open(out_file, "w") as f:
        f.write("episode,accuracy\n")
        for idx, acc in enumerate(all_acc):
            f.write(f"{idx},{acc}\n")
    print(f"✅ Saved CSV to {out_file}")

if __name__ == "__main__":
    main()
