import os
import subprocess

def main():
    print("=" * 56)
    print("KAN-SHOT Ablation Studies (3 Seeds)")
    print("=" * 56)
    print("Peringatan: Proses ini (terutama Fine-Tuning) akan memakan")
    print("waktu yang SANGAT LAMA karena mengeksekusi 3 seed berturut-turut!\n")
    
    input("Press Enter to continue...")

    for seed in [2021, 2022, 2023]:
        print("\n" + "=" * 40)
        print(f"1. Evaluasi Robustness KAN-SHOT (Seed {seed})")
        print("=" * 40)
        
        robustness_cmd = [
            "python", "scripts/eval_robustness.py",
            "--metric", f"checkpoints/seed{seed}/kanprotonet_cosine_nct.pth",
            "--manifest", f"episodes/lc5way_shot5_seed{seed}.json",
            "--episodes", "500"
        ]
        
        # Save output to text file
        output_file = f"results/robustness_seed{seed}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            subprocess.run(robustness_cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
            
        print("\n" + "=" * 40)
        print(f"2. Evaluasi Episodic Fine-Tuning FT-ALL (Seed {seed})")
        print("=" * 40)
        
        finetune_cmd = [
            "python", "scripts/eval_finetune.py",
            "--metric", f"checkpoints/seed{seed}/kanprotonet_cosine_nct.pth",
            "--manifest", f"episodes/lc5way_shot5_seed{seed}.json",
            "--output_dir", "results/multiseed",
            "--episodes", "500",
            "--ft_layers", "all"
        ]
        subprocess.run(finetune_cmd, check=True)

    print("\nSelesai! Seluruh hasil dari 3 seed telah disimpan di folder results/")

if __name__ == "__main__":
    main()
