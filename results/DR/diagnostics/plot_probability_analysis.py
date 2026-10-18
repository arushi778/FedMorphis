"""
Generates visualization of predicted probability distribution on GT lesion pixels vs Background
and component erasure breakdown.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    diag_dir = REPO_ROOT / "results" / "DR" / "diagnostics"
    
    # Data from audit_probability_distribution.py
    bin_labels = ['<0.01', '0.01-0.05', '0.05-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
    bin_pcts = [65.88, 2.50, 0.88, 1.04, 0.74, 0.60, 0.54, 0.58, 0.62, 0.78, 1.44, 24.40]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    
    # 1. Bar chart of predicted probability on GT pixels
    colors = ['crimson' if i < 7 else 'forestgreen' for i in range(len(bin_labels))]
    bars = ax1.bar(bin_labels, bin_pcts, color=colors, edgecolor='black', alpha=0.85)
    ax1.set_title("Predicted Probability Distribution on Ground-Truth Lesion Pixels\n(Demonstrating Extreme Bimodality: 65.9% at p < 0.01)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Predicted Sigmoid Probability Bin", fontweight="bold")
    ax1.set_ylabel("% of Ground-Truth Lesion Pixels", fontweight="bold")
    ax1.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.axvline(6.5, color='black', linestyle='--', linewidth=1.5, label='Threshold 0.50 Boundary')
    
    for bar, pct in zip(bars, bin_pcts):
        if pct > 1.0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.8, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.legend()
    
    # 2. Lesion Component Retention / Erasure by Type
    lesions = ['EX (Hard Exudates)', 'HE (Hemorrhages)', 'MA (Microaneurysms)', 'SE (Soft Exudates)', 'ALL LESIONS']
    retained = [91.7, 100.0, 90.5, 100.0, 94.8]
    erased = [8.3, 0.0, 9.5, 0.0, 5.2]
    
    x = np.arange(len(lesions))
    width = 0.45
    
    ax2.bar(x, retained, width, label='Retained Components', color='cornflowerblue', edgecolor='black')
    ax2.bar(x, erased, width, bottom=retained, label='Erased Components (512x512 Nearest)', color='salmon', edgecolor='black')
    
    ax2.set_title("Lesion Component Retention vs Erasure after 512x512 NEAREST Resize\n(9.5% of MAs and 8.3% of EXs Erased Completely)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Component Percentage (%)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(lesions, rotation=20, ha='right')
    ax2.set_ylim(0, 115)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')
    
    for i in range(len(lesions)):
        ax2.text(i, retained[i] / 2, f"{retained[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold')
        if erased[i] > 1.0:
            ax2.text(i, retained[i] + erased[i] / 2, f"{erased[i]:.1f}%", ha='center', va='center', color='black', fontweight='bold')

    plt.tight_layout()
    out_path = diag_dir / "val_probability_bimodality_and_component_erasure.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
