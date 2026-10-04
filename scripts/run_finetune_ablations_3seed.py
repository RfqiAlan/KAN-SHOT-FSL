import os
import subprocess

def main():
    print("=" * 56)
    print("KAN-SHOT Episodic Fine-Tuning Ablations (3 Seeds)")
    print("=" * 56)
    print("Peringatan: Proses ini akan memakan waktu SANGAT LAMA!\n")
    
    input("Press Enter to continue...")

    seeds = [2021, 2022, 2023]

    for seed in seeds:
        print("\n" + "=" * 40)
        print(f"Seed {seed}")
        print("=" * 40)

        print("5. FT-ALL ProtoNet [Seluruh Backbone, Head Tidak Ada]")
        cmd = [
            "python", "scripts/eval_finetune.py",
            "--metric", f"checkpoints/seed{seed}/kanprotonet_cosine_nct.pth",
            "--manifest", f"episodes/lc5way_shot5_seed{seed}.json",
            "--output_dir", "results/multiseed",
            "--episodes", "500",
            "--ft_layers", "all",
            "--override_method", "ProtoNet"
        ]
        subprocess.run(cmd, check=True)

    print("\nSelesai! Seluruh hasil dari 3 seed telah disimpan di folder results/multiseed/")

if __name__ == "__main__":
    main()
