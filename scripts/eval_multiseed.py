"""
Multi-Seed Evaluation Script.
Evaluates KANProtoNet, MLPProtoNet, and ProtoNet across 3 seeds (2021, 2022, 2023)
on all target-domain episode manifests.

Outputs:
  - Per-seed accuracy (terminal + CSV per seed)
  - Aggregated 3-seed results (mean ± 95% CI from 3000 episodes) saved to CSV

Usage:
    python scripts/eval_multiseed.py --protocol lc5way --shot 5
    python scripts/eval_multiseed.py --protocol crc_val --shot 5 --methods KANProtoNet ProtoNet
    python scripts/eval_multiseed.py --protocol lc5way --shot 1 --shot 5 --shot 10
"""
import os
import sys
import argparse
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kan_shot.methods import KANProtoNet, MLPProtoNet, ProtoNet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEEDS = [2021, 2022, 2023]


def load_image(img_path: Path, transform):
    img = Image.open(img_path).convert('RGB')
    return transform(img)


def load_backbone(backbone_path: str, device: torch.device):
    """Load frozen ResNet18 backbone."""
    model = models.resnet18(weights=None)
    model.fc = nn.Identity()
    raw = torch.load(backbone_path, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw)
    state = {k: v for k, v in state.items()
             if not k.startswith("fc.")
             and not k.endswith(("fc.weight", "fc.bias",
                                 "classifier.weight", "classifier.bias"))}
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model


def load_metric_method(method_name: str, ckpt_path: str, device: torch.device):
    """Load a trainable metric method (KANProtoNet or MLPProtoNet) from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    m_args = argparse.Namespace(**config)

    if method_name == "KANProtoNet":
        method = KANProtoNet(m_args)
    elif method_name == "MLPProtoNet":
        method = MLPProtoNet(m_args)
    else:
        raise ValueError(f"Unknown trainable method: {method_name}")

    method.load_state_dict(ckpt["method_state_dict"], strict=True)
    method = method.to(device)
    method.eval()
    return method


def load_protonet(device: torch.device, distance_metric="cosine"):
    """Load ProtoNet (no trainable metric params)."""
    m_args = argparse.Namespace(
        distance_metric=distance_metric,
        temperature=1.0,
        label_smoothing=0.0
    )
    method = ProtoNet(m_args)
    method = method.to(device)
    method.eval()
    return method


def evaluate_episodes(method, model, manifest_path: str, image_size: int,
                      device: torch.device, desc: str = ""):
    """Run evaluation on a single manifest, return per-episode accuracies."""
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    with open(manifest_path, "r") as f:
        episodes = json.load(f)

    accs = []
    with torch.no_grad():
        for ep in tqdm(episodes, desc=desc, leave=False):
            root = Path(ep["dataset_root"])
            label_map = ep["label_map"]
            method.n_way = ep["n_way"]

            support_imgs, support_labels = [], []
            for cls_name, rel_paths in ep["support"].items():
                local_label = label_map[cls_name]
                for rp in rel_paths:
                    support_imgs.append(load_image(root / rp, eval_transform))
                    support_labels.append(local_label)

            query_imgs, query_labels = [], []
            for q_item in ep["query"]:
                query_imgs.append(load_image(root / q_item["path"], eval_transform))
                query_labels.append(q_item["label"])

            x_s = torch.stack(support_imgs).unsqueeze(0).to(device)
            x_q = torch.stack(query_imgs).unsqueeze(0).to(device)
            y_s = torch.tensor(support_labels).unsqueeze(0).to(device)
            y_q = torch.tensor(query_labels).unsqueeze(0).to(device)

            _, preds_q = method(x_s=x_s, x_q=x_q, y_s=y_s, y_q=y_q, model=model)
            ep_acc = (preds_q == y_q).float().mean().item()
            accs.append(ep_acc)

    return np.array(accs)


def save_csv(accs: np.ndarray, path: str, extra_meta: dict = None):
    """Save per-episode accuracies + stats to CSV."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    mean_acc = np.mean(accs) * 100
    std_acc = np.std(accs) * 100
    ci95 = 1.96 * (std_acc / np.sqrt(len(accs)))

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if extra_meta:
            for k, v in extra_meta.items():
                writer.writerow([f"#{k}", v])
        writer.writerow(["episode_id", "accuracy"])
        for idx, a in enumerate(accs):
            writer.writerow([idx, f"{a:.4f}"])
        writer.writerow(["MEAN", f"{mean_acc:.2f}"])
        writer.writerow(["STD", f"{std_acc:.2f}"])
        writer.writerow(["CI95", f"{ci95:.2f}"])
        writer.writerow(["N_EPISODES", len(accs)])


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Seed Evaluation (3 seeds × N methods)"
    )
    parser.add_argument("--protocol", type=str, required=True,
                        choices=["lc5way", "crc_val", "lc_colon", "lc_lung"],
                        help="Evaluation protocol / dataset")
    parser.add_argument("--shot", type=int, nargs="+", default=[5],
                        help="Shot value(s) to evaluate (default: 5)")
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["KANProtoNet", "MLPProtoNet", "ProtoNet"],
                        help="Methods to evaluate")
    parser.add_argument("--backbone", type=str,
                        default=str(PROJECT_ROOT / "checkpoints" / "seed2021" / "resnet18_nct.pth"),
                        help="Path to backbone checkpoint")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help="Seeds to evaluate")
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--output_dir", type=str, default="results/multiseed",
                        help="Output directory for CSVs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load backbone once
    print(f"🔧 Loading backbone: {args.backbone}")
    model = load_backbone(args.backbone, device)

    for shot in args.shot:
        for method_name in args.methods:
            print(f"\n{'=' * 70}")
            print(f"📊 Evaluating: {method_name} | {args.protocol} | {shot}-shot")
            print(f"{'=' * 70}")

            all_seed_accs = {}  # seed -> np.array of per-episode accs

            for seed in args.seeds:
                manifest_path = str(
                    PROJECT_ROOT / "episodes" / f"{args.protocol}_shot{shot}_seed{seed}.json"
                )
                if not os.path.exists(manifest_path):
                    print(f"⚠️  Manifest not found: {manifest_path}, skipping seed {seed}")
                    continue

                # Load method
                if method_name == "ProtoNet":
                    method = load_protonet(device)
                else:
                    # Trainable methods: load seed-specific checkpoint
                    ckpt_name = f"{method_name.lower()}_cosine_nct.pth"
                    ckpt_path = str(
                        PROJECT_ROOT / "checkpoints" / f"seed{seed}" / ckpt_name
                    )
                    if not os.path.exists(ckpt_path):
                        print(f"⚠️  Checkpoint not found: {ckpt_path}, skipping seed {seed}")
                        continue
                    method = load_metric_method(method_name, ckpt_path, device)

                # Evaluate
                desc = f"seed={seed}"
                accs = evaluate_episodes(
                    method, model, manifest_path, args.image_size, device, desc
                )
                all_seed_accs[seed] = accs

                # Print per-seed results
                mean_acc = np.mean(accs) * 100
                std_acc = np.std(accs) * 100
                ci95 = 1.96 * (std_acc / np.sqrt(len(accs)))
                print(f"   Seed {seed}: {mean_acc:.2f}% ± {ci95:.2f}% "
                      f"(1000 ep, std={std_acc:.2f}%)")

                # Save per-seed CSV
                seed_csv = os.path.join(
                    args.output_dir,
                    f"{method_name.lower()}_{args.protocol}_shot{shot}_seed{seed}.csv"
                )
                save_csv(accs, seed_csv, extra_meta={
                    "method": method_name,
                    "protocol": args.protocol,
                    "shot": shot,
                    "seed": seed,
                })

            # ── Aggregation across seeds ──
            if len(all_seed_accs) == 0:
                print(f"   ❌ No seeds evaluated for {method_name}")
                continue

            # Concatenate all episodes across seeds
            aggregated = np.concatenate(list(all_seed_accs.values()))
            agg_mean = np.mean(aggregated) * 100
            agg_std = np.std(aggregated) * 100
            agg_ci95 = 1.96 * (agg_std / np.sqrt(len(aggregated)))

            # Also compute seed-level mean (mean of per-seed means)
            seed_means = [np.mean(v) * 100 for v in all_seed_accs.values()]
            seed_level_mean = np.mean(seed_means)
            seed_level_std = np.std(seed_means)

            print(f"\n   {'─' * 50}")
            print(f"   📈 AGGREGATED ({len(all_seed_accs)} seeds × 1000 ep "
                  f"= {len(aggregated)} episodes):")
            print(f"   Episode-level:  {agg_mean:.2f}% ± {agg_ci95:.2f}% (95% CI)")
            print(f"   Seed-level:     {seed_level_mean:.2f}% ± {seed_level_std:.2f}% (across seeds)")

            # Save aggregated CSV
            agg_csv = os.path.join(
                args.output_dir,
                f"{method_name.lower()}_{args.protocol}_shot{shot}_aggregated.csv"
            )
            save_csv(aggregated, agg_csv, extra_meta={
                "method": method_name,
                "protocol": args.protocol,
                "shot": shot,
                "seeds": str(list(all_seed_accs.keys())),
                "n_seeds": len(all_seed_accs),
                "seed_level_mean": f"{seed_level_mean:.2f}",
                "seed_level_std": f"{seed_level_std:.2f}",
            })
            print(f"   💾 Saved: {agg_csv}")

    # ── Final Summary Table ──
    print(f"\n{'=' * 70}")
    print("📋 FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Method':<16} {'Protocol':<10} {'Shot':<6} "
          f"{'Mean%':<10} {'±CI95':<10} {'Seeds':<6}")
    print("-" * 58)

    for shot in args.shot:
        for method_name in args.methods:
            agg_csv = os.path.join(
                args.output_dir,
                f"{method_name.lower()}_{args.protocol}_shot{shot}_aggregated.csv"
            )
            if os.path.exists(agg_csv):
                # Read back the aggregated stats
                with open(agg_csv, "r") as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    meta = {r[0]: r[1] for r in rows if r[0].startswith("#")}
                    stats = {r[0]: r[1] for r in rows if r[0] in ("MEAN", "CI95", "N_EPISODES")}
                    n_seeds = meta.get("#n_seeds", "?")
                    print(f"{method_name:<16} {args.protocol:<10} {shot:<6} "
                          f"{stats.get('MEAN', '?'):<10} {stats.get('CI95', '?'):<10} "
                          f"{n_seeds:<6}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
