#!/usr/bin/env python3
"""
Evaluasi Multi-Seed untuk Trainable Metric Modules.
Membaca checkpoint dari masing-masing seed (2021, 2022, 2023)
dan mengevaluasinya pada manifest episode yang sesuai.
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
import csv

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kan_shot.methods import KANProtoNet, MLPProtoNet


def load_image(img_path: Path, transform):
    img = Image.open(img_path).convert('RGB')
    return transform(img)


def evaluate_single_seed(seed, method_name, checkpoint_dir, manifest_prefix, data_root, image_size, device):
    """
    Evaluates one seed and returns the list of accuracies (length=1000).
    """
    # Expected paths
    # checkpoint_path e.g. checkpoints/seed2021/kanprotonet_best_nct.pth
    # Or just search for the latest checkpoint
    import glob
    ckpt_pattern = os.path.join(checkpoint_dir, f"seed{seed}", f"{method_name.lower()}*.pth")
    ckpts = glob.glob(ckpt_pattern)
    if not ckpts:
        raise FileNotFoundError(f"Tidal menemukan checkpoint di {ckpt_pattern}")
    metric_path = ckpts[0] # Ambil yang pertama jika ada banyak
    
    manifest_path = f"{manifest_prefix}_seed{seed}.json"
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest tidak ditemukan: {manifest_path}")

    print(f"\n[{method_name} - SEED {seed}] Loading checkpoint: {metric_path}")
    
    ckpt = torch.load(metric_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    m_args = argparse.Namespace(**config)
    
    backbone_path = ckpt.get("backbone_checkpoint", "checkpoints/seed2021/resnet18_nct.pth")
    if not os.path.exists(backbone_path):
        print(f"Warning: Backbone di {backbone_path} tidak ditemukan, mencoba load langsung ke model...")
        
    model = models.resnet18(weights=None)
    model.fc = nn.Identity()
    
    if os.path.exists(backbone_path):
        raw_backbone = torch.load(backbone_path, map_location="cpu", weights_only=False)
        state_b = raw_backbone.get("state_dict", raw_backbone)
        state_b = {k: v for k, v in state_b.items() if not k.startswith("fc.") and not k.endswith(("fc.weight", "fc.bias", "classifier.weight", "classifier.bias"))}
        model.load_state_dict(state_b, strict=False)
    
    model = model.to(device)
    model.eval()

    if config["method"] == "KANProtoNet":
        method = KANProtoNet(m_args)
    elif config["method"] == "MLPProtoNet":
        method = MLPProtoNet(m_args)
    else:
        # Default fallback (e.g. ProtoNet)
        pass # To do if needed
        
    method.load_state_dict(ckpt["method_state_dict"], strict=True)
    method = method.to(device)
    method.eval()

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with open(manifest_path, "r") as f:
        episodes = json.load(f)

    print(f"[{method_name} - SEED {seed}] Evaluating {len(episodes)} episodes...")
    accs = []
    with torch.no_grad():
        for ep in tqdm(episodes, desc=f"Seed {seed}", leave=False):
            # Resolve data path dynamically using local data_root
            original_root = Path(ep["dataset_root"])
            # Fallback to local data_root
            local_root = Path(data_root) / original_root.name
            
            label_map = ep["label_map"]
            
            m_args.n_way = ep["n_way"]
            method.n_way = ep["n_way"]

            support_imgs = []
            support_labels = []
            for cls_name, rel_paths in ep["support"].items():
                local_label = label_map[cls_name]
                for rp in rel_paths:
                    img_path = local_root / rp
                    support_imgs.append(load_image(img_path, eval_transform))
                    support_labels.append(local_label)

            query_imgs = []
            query_labels = []
            for q_item in ep["query"]:
                q_path = local_root / q_item["path"]
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

    print(f"   => Seed {seed} Accuracy: {mean_acc:.2f}% ± {ci95:.2f}% (95% CI)")
    return accs.tolist()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Seed Trainable Metric Modules")
    parser.add_argument("--method", type=str, default="KANProtoNet")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Base directory containing seedXXXX folders")
    parser.add_argument("--manifest_prefix", type=str, required=True, help="Prefix of manifest (e.g. 'episodes/lc5way_shot5')")
    parser.add_argument("--data_root", type=str, default="data", help="Local directory containing dataset folders")
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--output", type=str, default="results_multiseed.csv", help="Path to output CSV")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2021, 2022, 2023], help="Seeds to evaluate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    print("="*60)
    print(f"MULTI-SEED EVALUATION: {args.method}")
    print(f"Manifest Prefix: {args.manifest_prefix}")
    print("="*60)

    all_accs = []
    
    for seed in args.seeds:
        try:
            accs = evaluate_single_seed(
                seed=seed,
                method_name=args.method,
                checkpoint_dir=args.checkpoint_dir,
                manifest_prefix=args.manifest_prefix,
                data_root=args.data_root,
                image_size=args.image_size,
                device=device
            )
            all_accs.extend(accs)
        except Exception as e:
            print(f"❌ Gagal mengevaluasi seed {seed}: {e}")
            
    if not all_accs:
        print("❌ Tidak ada data yang dievaluasi.")
        return

    # Total agregasi 3.000 episode
    all_accs = np.array(all_accs)
    mean_acc = np.mean(all_accs) * 100
    std_acc = np.std(all_accs) * 100
    ci95 = 1.96 * (std_acc / np.sqrt(len(all_accs)))

    print("\n" + "="*60)
    print(f"HASIL AKHIR GABUNGAN ({len(all_accs)} Episode):")
    print(f"Mean Accuracy: {mean_acc:.2f}% ± {ci95:.2f}% (95% CI)")
    print("="*60 + "\n")

    # Save CSV
    import csv
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode_id", "accuracy"])
        for idx, a in enumerate(all_accs):
            writer.writerow([idx, f"{a:.4f}"])
        writer.writerow(["MEAN", f"{mean_acc:.2f}"])
        writer.writerow(["STD", f"{std_acc:.2f}"])
        writer.writerow(["CI95", f"{ci95:.2f}"])
        writer.writerow(["TOTAL_EPISODES", len(all_accs)])

    print(f"✅ Saved CSV to {args.output}")

if __name__ == "__main__":
    main()
