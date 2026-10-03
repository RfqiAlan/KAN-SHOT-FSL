#!/usr/bin/env python3
"""
Wrapper script untuk menjalankan Phase B Training pada 3 seed (2021, 2022, 2023)
secara otomatis. Mengasumsikan dataset berada di dalam folder `data`.
"""

import os
import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Phase B Training")
    parser.add_argument("--method", type=str, default="KANProtoNet", choices=["KANProtoNet", "MLPProtoNet", "ProtoNet"])
    parser.add_argument("--backbone", type=str, default="checkpoints/seed2021/resnet18_nct.pth", help="Path to frozen backbone")
    parser.add_argument("--data_root", type=str, default="data", help="Root directory for dataset")
    parser.add_argument("--train_iter", type=int, default=30000, help="Number of training iterations per seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2021, 2022, 2023], help="Seeds to evaluate")
    
    # Allow passing extra arguments directly to run_phase_b_metric.py
    args, unknown = parser.parse_known_args()
    
    if not os.path.exists(args.backbone):
        print(f"❌ Error: Backbone {args.backbone} tidak ditemukan!")
        print("Pastikan Anda sudah menjalankan Phase A setidaknya 1 kali.")
        sys.exit(1)
        
    for seed in args.seeds:
        output_dir = f"checkpoints/seed{seed}"
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [
            sys.executable, "scripts/run_phase_b_metric.py",
            "--method", args.method,
            "--backbone", args.backbone,
            "--data_root", args.data_root,
            "--seed", str(seed),
            "--output_dir", output_dir,
            "--train_iter", str(args.train_iter)
        ] + unknown
        
        print("\n" + "="*60)
        print(f"🚀 MEMULAI PELATIHAN PHASE B UNTUK SEED {seed}")
        print("="*60)
        print("Command:", " ".join(cmd))
        
        # Jalankan skrip Phase B
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Pelatihan gagal pada seed {seed}. Menghentikan proses.")
            sys.exit(e.returncode)
            
    print("\n✅ PELATIHAN MULTI-SEED (2021, 2022, 2023) SELESAI!")
    print("Checkpoint berhasil disimpan di folder checkpoints/seedXXXX/")

if __name__ == "__main__":
    main()
