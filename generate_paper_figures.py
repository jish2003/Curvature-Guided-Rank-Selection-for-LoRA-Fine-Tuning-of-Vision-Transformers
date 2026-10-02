"""
Generates Figure 3 (Global CGRS rank trajectories) and Figure 4 (per-layer
final-rank heatmap) directly from each dataset's all_results_complete.json.
No hardcoded numbers -- everything is read live from the JSON files.

Run this from the REPOSITORY ROOT (the folder containing CIFAR100_Rebuild/,
SVHN_Rebuild/, and Flowers102_Rebuild/ as siblings), e.g.:

    cd Curvature-Guided-Rank-Selection-for-LoRA-Fine-Tuning-of-Vision-Transformers
    python generate_paper_figures.py

Outputs:
    figures/fig3_rank_trajectory.png
    figures/fig4_module_heatmap.png
"""
import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = {
    "CIFAR-100": {"json_path": "CIFAR100_Rebuild/all_results_complete.json", "total_steps": 8439},
    "SVHN": {"json_path": "SVHN_Rebuild/all_results_complete.json", "total_steps": 12750},
    "Flowers-102": {"json_path": "Flowers102_Rebuild/all_results_complete.json", "total_steps": 2300},
}

THRESHOLD_RUNS = {
    "aggressive": "CGRS_Global_tau_aggressive",
    "moderate": "CGRS_Global_tau_moderate",
    "conservative": "CGRS_Global_tau_conservative",
}
THRESHOLD_COLORS = {"aggressive": "#d62728", "moderate": "#ff7f0e", "conservative": "#2ca02c"}
R_INIT_GLOBAL = 16

# Curated module-wise runs for Figure 4 -- edit these if you want different examples
HEATMAP_CASES = [
    ("SVHN", "PL_C1", "SVHN PL-C1 (heterogeneous)"),
    ("CIFAR-100", "PL_C3_Live", "CIFAR-100 PL-C3-Live (heterogeneous)"),
    ("Flowers-102", "PL_C3_Live", "Flowers-102 PL-C3-Live (uniform collapse)"),
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_step_function(rank_changes, r_init, total_steps):
    steps = [0]
    ranks = [r_init]
    cur_r = r_init
    for change in sorted(rank_changes, key=lambda c: c["step"]):
        steps.append(change["step"])
        ranks.append(cur_r)          # hold old rank right up to the change
        steps.append(change["step"])
        ranks.append(change["new_r"])
        cur_r = change["new_r"]
    steps.append(total_steps)
    ranks.append(cur_r)
    steps_pct = [100.0 * s / total_steps for s in steps]
    return steps_pct, ranks


def make_fig3_rank_trajectory():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, (dataset_name, cfg) in zip(axes, DATASETS.items()):
        data = load_json(cfg["json_path"])
        phase3 = data.get("phase3", {})
        for label, run_key in THRESHOLD_RUNS.items():
            if run_key not in phase3:
                continue
            run = phase3[run_key]
            rank_changes = run.get("rank_changes", [])
            steps_pct, ranks = build_step_function(rank_changes, R_INIT_GLOBAL, cfg["total_steps"])
            ax.step(steps_pct, ranks, where="post", label=label,
                     color=THRESHOLD_COLORS[label], linewidth=1.8)
        ax.axvspan(0, 15, alpha=0.12, color="gray")
        ax.set_title(dataset_name, fontsize=11)
        ax.set_xlabel("Training progress (%)")
        ax.set_yticks([3, 5, 6, 10, 12, 16, 30, 52, 64])
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Global LoRA rank")
    axes[-1].legend(fontsize=8, loc="lower right", title="threshold")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig3_rank_trajectory.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def make_fig4_module_heatmap():
    rows = []
    row_labels = []
    for dataset_name, run_key, display_label in HEATMAP_CASES:
        cfg = DATASETS[dataset_name]
        data = load_json(cfg["json_path"])
        phase4 = data.get("phase4", {})
        if run_key not in phase4:
            print(f"WARNING: {run_key} not found for {dataset_name}, skipping.")
            continue
        final_ranks = phase4[run_key]["final_ranks"]
        row = [final_ranks[str(i)] for i in range(12)]
        rows.append(row)
        row_labels.append(display_label)

    matrix = np.array(rows)
    fig, ax = plt.subplots(figsize=(9, 2.6))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=3, vmax=64)
    ax.set_xticks(range(12))
    ax.set_xticklabels([f"L{i}" for i in range(12)], fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                     color="white" if matrix[i, j] < 40 else "black", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Final rank", fontsize=8)
    ax.set_xlabel("Transformer layer")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig4_module_heatmap.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    make_fig3_rank_trajectory()
    make_fig4_module_heatmap()
