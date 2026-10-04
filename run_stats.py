import os
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from evaluation.statistical_tests import pairwise_comparison, format_report

def main():
    # 1. Compare LC5way 5-shot frozen
    proto = pd.read_csv("results/proto_lc5way_5shot.csv")["accuracy"].values
    kan = pd.read_csv("results/kan_distance_lc5way_5shot.csv")["accuracy"].values
    
    # We also have mlp results coming soon, but we can do Proto vs KAN for now.
    
    models = {
        "ProtoNet": proto,
        "KAN-Distance": kan
    }
    
    report = pairwise_comparison(models, "LC 5-way 5-shot (Frozen)")
    print(format_report(report))
    
    # Write to a file
    with open("results/wilcoxon_report.txt", "w", encoding="utf-8") as f:
        f.write(format_report(report))
        
if __name__ == "__main__":
    main()
