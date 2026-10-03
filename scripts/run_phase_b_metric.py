"""
Phase B: Metric Module Training on NCT Episodes with Frozen ResNet18 Backbone.
Trains KANProtoNet or MLPProtoNet on few-shot episodes from nct_train.
Selects best checkpoint based on nct_val episodes.
"""
import os
import sys

# Add root directory to sys.path to find kan_shot
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm

from kan_shot.methods import KANProtoNet, MLPProtoNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


from torch.utils.data import Sampler, DataLoader

class EpisodicBatchSampler(Sampler):
    def __init__(self, dataset_labels, n_way: int, k_shot: int, q_query: int, num_episodes: int):
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_episodes = num_episodes

        self.class_indices = {}
        for idx, label in enumerate(dataset_labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        self.classes = sorted(list(self.class_indices.keys()))
        assert len(self.classes) >= n_way, f"Dataset only has {len(self.classes)} classes, expected at least {n_way}"

    def __iter__(self):
        for _ in range(self.num_episodes):
            selected_classes = random.sample(self.classes, self.n_way)
            support_idx = []
            query_idx = []
            for cls_id in selected_classes:
                sampled = random.sample(self.class_indices[cls_id], self.k_shot + self.q_query)
                support_idx.extend(sampled[:self.k_shot])
                query_idx.extend(sampled[self.k_shot:])
            # Yield indices: all supports first, then all queries
            yield support_idx + query_idx

    def __len__(self):
        return self.num_episodes


def load_and_freeze_backbone(checkpoint_path: str, device: torch.device):
    model = models.resnet18(weights=None)
    model.fc = nn.Identity()

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw)
    state = {k: v for k, v in state.items() if not k.startswith("fc.") and not k.endswith(("fc.weight", "fc.bias", "classifier.weight", "classifier.bias"))}
    missing, unexpected = model.load_state_dict(state, strict=False)

    allowed_missing = {"fc.weight", "fc.bias"}
    missing_set = set(missing)
    unexpected_missing = missing_set - allowed_missing
    assert not unexpected, f"Unexpected keys in checkpoint: {unexpected}"
    assert not unexpected_missing, f"Unexpected missing keys: {unexpected_missing}"

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    assert all(not p.requires_grad for p in model.parameters()), "Backbone tidak terbekukan!"
    print("✅ Backbone successfully loaded and frozen.")
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="Phase B: Metric Module Training on Episodes")
    parser.add_argument("--method", type=str, choices=["KANProtoNet", "MLPProtoNet"], required=True)
    parser.add_argument("--distance_metric", type=str, choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    
    # KAN args
    parser.add_argument("--kan_mode", type=str, default="transform", choices=["transform", "distance", "dual"])
    parser.add_argument("--kan_hidden", type=int, default=None)
    parser.add_argument("--kan_dropout", type=float, default=0.15)
    parser.add_argument("--kan_prenorm", action="store_true", help="L2 normalize features before KAN transform")
    parser.add_argument("--lambda_spline", type=float, default=0.01, help="L1 regularization for KAN spline weights")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (in val checks)")
    
    parser.add_argument("--backbone", type=str, required=True, help="Path to resnet18_nct.pth")
    parser.add_argument("--data_root", type=str, default="data/raw")
    parser.add_argument("--train_source", type=str, default="nct_train")
    parser.add_argument("--val_source", type=str, default="nct_val")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--train_iter", type=int, default=30000)
    parser.add_argument("--val_iter", type=int, default=250)
    parser.add_argument("--val_freq", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--n_way", type=int, default=9)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--q_query", type=int, default=15)
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--output_dir", type=str, default="checkpoints/seed2021")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Backbone
    model = load_and_freeze_backbone(args.backbone, device)

    # 2. Metric Method
    if args.method == "KANProtoNet":
        method = KANProtoNet(args)
    else:
        method = MLPProtoNet(args)
    method = method.to(device)

    trainable_params = [p for p in method.parameters() if p.requires_grad]
    assert len(trainable_params) > 0, "Method tidak memiliki trainable parameter!"
    print(f"✅ {args.method} initialized. Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Verify optimizer does not touch backbone
    model_ids = {id(p) for p in model.parameters()}
    opt_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert opt_ids.isdisjoint(model_ids), "Optimizer contains backbone parameters!"

    # 3. Data Pipelines
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(args.data_root, args.train_source)
    val_dir = os.path.join(args.data_root, args.val_source)

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)

    train_sampler = EpisodicBatchSampler(train_dataset.targets, args.n_way, args.k_shot, args.q_query, args.train_iter)
    val_sampler = EpisodicBatchSampler(val_dataset.targets, args.n_way, args.k_shot, args.q_query, args.val_iter)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=4, pin_memory=True)

    best_val_acc = 0.0
    patience_counter = 0
    model_save_name = f"{args.method.lower()}_{args.distance_metric}_nct.pth"
    save_path = os.path.join(args.output_dir, model_save_name)

    # Config to save in checkpoint
    checkpoint_config = {
        "method": args.method,
        "distance_metric": args.distance_metric,
        "temperature": args.temperature,
        "label_smoothing": args.label_smoothing,
        "feat_dim": getattr(args, 'feat_dim', 512),
        "kan_out_dim": getattr(args, 'kan_out_dim', 512),
    }
    if args.method == "KANProtoNet":
        checkpoint_config["kan_mode"] = args.kan_mode
        checkpoint_config["kan_hidden"] = args.kan_hidden
        checkpoint_config["kan_prenorm"] = getattr(args, 'kan_prenorm', False)

    # 4. Training Loop
    method.train()
    pbar = tqdm(enumerate(train_loader, 1), total=args.train_iter, desc="Training Episodes")
    
    first_step_checked = False

    # Pre-compute labels (they are identical for every generated episode)
    y_s = torch.arange(args.n_way).repeat_interleave(args.k_shot).unsqueeze(0).to(device)
    y_q = torch.arange(args.n_way).repeat_interleave(args.q_query).unsqueeze(0).to(device)
    num_support = args.n_way * args.k_shot

    for step, (batch_imgs, _) in pbar:
        batch_imgs = batch_imgs.to(device)
        
        # Split batch into support and query
        x_s = batch_imgs[:num_support].unsqueeze(0)
        x_q = batch_imgs[num_support:].unsqueeze(0)

        before_params = {n: p.detach().clone() for n, p in method.named_parameters() if p.requires_grad}

        optimizer.zero_grad()
        loss, preds_q = method(x_s=x_s, x_q=x_q, y_s=y_s, y_q=y_q, model=model)
        loss = loss.mean()

        # Add Spline L1 Regularization if using KAN
        if args.method == "KANProtoNet" and args.lambda_spline > 0:
            spline_reg = 0.0
            from kan_shot.kan.kan_layer import KANLinear
            for module in method.modules():
                if isinstance(module, KANLinear):
                    spline_reg += module.spline_weight.abs().mean()
            loss = loss + args.lambda_spline * spline_reg

        loss.backward()

        if not first_step_checked:
            assert all(p.grad is None for p in model.parameters()), "Backbone received gradients during Phase B!"
            grads = {n: p.grad for n, p in method.named_parameters() if p.requires_grad and p.grad is not None}
            assert len(grads) > 0, "Method received no gradients!"

            print(f"\n📊 Gradient diagnostics ({len(grads)} param groups with gradients):")
            for n, g in list(grads.items()):
                print(f"   {n}: grad_norm={g.norm().item():.6f}, grad_mean={g.mean().item():.8f}")

            optimizer.step()

            changed = [not torch.equal(before_params[n], p.detach()) for n, p in method.named_parameters() if p.requires_grad]
            if any(changed):
                print("✅ Sanity check passed: Backbone frozen, method parameters actively update.")
            else:
                print("⚠️  Warning: Parameter values did not change after first step (may be normal for initial state).")
            first_step_checked = True
        else:
            optimizer.step()

        acc = (preds_q == y_q).float().mean().item()
        pbar.set_postfix({"Loss": f"{loss.item():.12f}", "Acc": f"{acc:.4f}"})

        if step % args.val_freq == 0:
            method.eval()
            val_accs = []
            with torch.no_grad():
                for val_batch_imgs, _ in val_loader:
                    val_batch_imgs = val_batch_imgs.to(device)
                    vx_s = val_batch_imgs[:num_support].unsqueeze(0)
                    vx_q = val_batch_imgs[num_support:].unsqueeze(0)

                    _, v_preds_q = method(x_s=vx_s, x_q=vx_q, y_s=y_s, y_q=y_q, model=model)
                    v_acc = (v_preds_q == y_q).float().mean().item()
                    val_accs.append(v_acc)

            mean_val_acc = np.mean(val_accs)
            print(f"\n[Step {step}] NCT-Val Acc ({args.val_iter} ep): {mean_val_acc:.4f}")

            if mean_val_acc > best_val_acc:
                best_val_acc = mean_val_acc
                torch.save({
                    "step": step,
                    "method_state_dict": method.state_dict(),
                    "config": checkpoint_config,
                    "val_acc": best_val_acc,
                    "train_seed": args.seed,
                    "backbone_checkpoint": args.backbone
                }, save_path)
                print(f"⭐ Saved new best metric module to {save_path} (Val Acc = {best_val_acc:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Patience: {patience_counter}/{args.patience}")
                if patience_counter >= args.patience:
                    print(f"🛑 Early stopping triggered after {step} steps. Best Val Acc: {best_val_acc:.4f}")
                    break

            method.train()

    print(f"✅ Phase B Training finished for {args.method} ({args.distance_metric}). Best NCT-Val Acc = {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
