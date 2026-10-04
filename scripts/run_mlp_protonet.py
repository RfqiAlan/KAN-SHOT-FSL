import os
import subprocess

def main():
    seed = 2021
    print("=" * 40)
    print(f"Training MLPProtoNet - Seed {seed}")
    print("=" * 40)

    train_cmd = [
        "python", "scripts/run_phase_b_metric.py",
        "--method", "MLPProtoNet",
        "--backbone", f"checkpoints/seed{seed}/resnet18_nct.pth",
        "--data_root", "data/raw",
        "--train_source", "nct_train",
        "--val_source", "nct_val",
        "--seed", str(seed),
        "--train_iter", "30000",
        "--val_iter", "250",
        "--val_freq", "1000",
        "--lr", "0.0005",
        "--output_dir", f"checkpoints/seed{seed}"
    ]
    subprocess.run(train_cmd, check=True)

    print("\n" + "=" * 40)
    print("Evaluating MLPProtoNet on Near-Domain (CRC-VAL 5-Shot)")
    print("=" * 40)

    eval_cmd = [
        "python", "scripts/eval_target.py",
        "--metric", f"checkpoints/seed{seed}/mlpprotonet_cosine_nct.pth",
        "--manifest", f"episodes/crc_val_shot5_seed{seed}.json",
        "--output", f"results/mlp_train{seed}_crc_val_shot5_eval{seed}.csv"
    ]
    subprocess.run(eval_cmd, check=True)

    print("\nDone! Results saved to", f"results/mlp_train{seed}_crc_val_shot5_eval{seed}.csv")

if __name__ == "__main__":
    main()
