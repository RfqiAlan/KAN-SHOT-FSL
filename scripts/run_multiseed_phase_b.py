"""
Multi-Seed Phase B Training Script.
Trains KANProtoNet and MLPProtoNet on 3 seeds (2021, 2022, 2023) using a shared
frozen backbone (seed 2021). Each seed produces a separate metric checkpoint under
checkpoints/seed{SEED}/.

Usage:
    python scripts/run_multiseed_phase_b.py --data_root data/raw
    python scripts/run_multiseed_phase_b.py --data_root data/raw --seeds 2022 2023   # skip 2021
    python scripts/run_multiseed_phase_b.py --data_root data/raw --methods KANProtoNet  # only KAN
"""
import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Default backbone: always seed2021 (single Phase-A run)
DEFAULT_BACKBONE = str(PROJECT_ROOT / "checkpoints" / "seed2021" / "resnet18_nct.pth")
SEEDS = [2021, 2022, 2023]
METHODS = ["KANProtoNet", "MLPProtoNet"]


def run_phase_b(method: str, seed: int, backbone: str, data_root: str,
                train_iter: int, extra_args: list):
    """Launch run_phase_b_metric.py as a subprocess."""
    output_dir = str(PROJECT_ROOT / "checkpoints" / f"seed{seed}")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, str(SCRIPT_DIR / "run_phase_b_metric.py"),
        "--method", method,
        "--backbone", backbone,
        "--seed", str(seed),
        "--data_root", data_root,
        "--output_dir", output_dir,
        "--train_iter", str(train_iter),
    ]
    cmd.extend(extra_args)

    print("\n" + "=" * 70)
    print(f"🚀 Phase B: {method} | seed={seed}")
    print(f"   backbone  = {backbone}")
    print(f"   output    = {output_dir}")
    print(f"   data_root = {data_root}")
    print("=" * 70 + "\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    status = "✅ SUCCESS" if result.returncode == 0 else "❌ FAILED"
    print(f"\n{status}: {method} seed={seed} ({elapsed:.0f}s)")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Seed Phase B Training (3 seeds × 2 methods)"
    )
    parser.add_argument(
        "--backbone", type=str, default=DEFAULT_BACKBONE,
        help="Path to frozen backbone checkpoint (default: seed2021)"
    )
    parser.add_argument(
        "--data_root", type=str, default="data/raw",
        help="Root folder containing nct_train and nct_val"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=SEEDS,
        help="List of seeds to train (default: 2021 2022 2023)"
    )
    parser.add_argument(
        "--methods", type=str, nargs="+", default=METHODS,
        choices=METHODS,
        help="List of methods to train (default: KANProtoNet MLPProtoNet)"
    )
    parser.add_argument(
        "--train_iter", type=int, default=30000,
        help="Number of training episodes per run"
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip training if checkpoint already exists"
    )
    args, extra = parser.parse_known_args()

    # Validate backbone exists
    if not os.path.exists(args.backbone):
        print(f"❌ Backbone not found: {args.backbone}")
        print("   Run Phase A first: python scripts/run_phase_a.py")
        sys.exit(1)

    results = []
    total = len(args.seeds) * len(args.methods)
    current = 0

    for seed in args.seeds:
        for method in args.methods:
            current += 1
            ckpt_name = f"{method.lower()}_cosine_nct.pth"
            ckpt_path = PROJECT_ROOT / "checkpoints" / f"seed{seed}" / ckpt_name

            if args.skip_existing and ckpt_path.exists():
                print(f"\n⏭️  [{current}/{total}] Skipping {method} seed={seed} "
                      f"(checkpoint exists: {ckpt_path})")
                results.append((method, seed, "SKIPPED"))
                continue

            print(f"\n📋 [{current}/{total}] Starting {method} seed={seed}")
            rc = run_phase_b(
                method=method,
                seed=seed,
                backbone=args.backbone,
                data_root=args.data_root,
                train_iter=args.train_iter,
                extra_args=extra,
            )
            results.append((method, seed, "OK" if rc == 0 else f"FAIL(rc={rc})"))

    # Summary
    print("\n" + "=" * 70)
    print("📊 MULTI-SEED TRAINING SUMMARY")
    print("=" * 70)
    print(f"{'Method':<16} {'Seed':<8} {'Status':<12}")
    print("-" * 36)
    for method, seed, status in results:
        print(f"{method:<16} {seed:<8} {status:<12}")
    print("=" * 70)

    # Check all checkpoints
    print("\n📁 Checkpoint inventory:")
    for seed in args.seeds:
        seed_dir = PROJECT_ROOT / "checkpoints" / f"seed{seed}"
        if seed_dir.exists():
            files = list(seed_dir.glob("*.pth"))
            print(f"   seed{seed}/: {', '.join(f.name for f in files) if files else '(empty)'}")
        else:
            print(f"   seed{seed}/: (directory not found)")


if __name__ == "__main__":
    main()
