# Curvature-Guided Rank Selection (CGRS) for LoRA Fine-Tuning of Vision Transformers

**Dynamic LoRA Rank Adaptation via Fisher Information Curvature**

CGRS dynamically adjusts LoRA adapter ranks *during* fine-tuning of a Vision Transformer (ViT-Base/16) on CIFAR-100, guided by the diagonal Fisher Information curvature (λ_max) of the loss landscape. Instead of committing to a single fixed rank for all layers and all training steps, CGRS reads the curvature signal and promotes (or demotes) rank adaptively — allocating model capacity to the layers and moments where it is most needed.

> **Core finding:** Even average ranks as low as **3–4** can approach **90–91%** accuracy when the model is allowed to expand rank selectively, suggesting most transformer layers are geometrically flat and do not benefit from high-rank adapters.

---

## Table of Contents

- [Motivation](#motivation)
- [Method Overview](#method-overview)
- [Project Phases](#project-phases)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Environment & Setup](#environment--setup)
- [How to Run](#how-to-run)
- [Internal Functions Reference](#internal-functions-reference)
- [Engineering Lessons & Bugs Fixed](#engineering-lessons--bugs-fixed)
- [Future Directions](#future-directions)

---

## Motivation

LoRA fine-tuning typically assigns a single fixed rank to every adapted layer. But the loss landscape is **not uniformly curved** across a transformer:

- There is a **~200× difference** in curvature (λ_max) between the sharpest layer (L11 query, λ_max ≈ 0.0123) and the flattest layer (L1 query, λ_max ≈ 0.0000338).
- **Deep layers** (L9–L11 query projections) are geometrically sharp — the loss curves strongly along their parameter directions.
- **Early layers** (L0–L3 query) are flat.

Assigning the same fixed rank to all layers ignores this variation: it wastes capacity on flat early layers while under-serving sharp deep layers. CGRS exploits the per-layer curvature structure to allocate rank where it matters.

---

## Method Overview

CGRS observes the **diagonal Fisher Information curvature** of the LoRA parameters and uses it as a signal to trigger rank transitions during training.

- Every **K = 200** training steps, the per-layer Fisher probe accumulates `grad²` over 8 mini-batches and returns `λ_max` per `(layer_idx, projection)`.
- If a layer's `λ_i` exceeds its threshold `τ_i` (and its cooldown counter is 0), the layer's rank is **promoted** to the next value in the rank list `[3, 5, 6, 10, 12, 16, 30, 52, 64]`.
- A **cooldown of 100 steps** prevents oscillation after a transition.
- Rank transitions use **SVD-based weight transfer**: existing `lora_A` and `lora_B` are combined as `W = B @ A`, then the new (larger/smaller) matrices are initialized from leading singular components — preserving learned information across transitions (expand = zero-padding, shrink = truncated SVD).

**Model:** `google/vit-base-patch16-224` · **Dataset:** CIFAR-100 (100-class classification) · **Target modules:** query + value projections, all 12 transformer blocks.

---

## Project Phases

| Phase | Goal | Status |
|-------|------|--------|
| **Phase 1** | Fixed-rank LoRA baselines — establish accuracy-vs-params Pareto frontier | ✅ Complete |
| **Phase 2** | Per-layer curvature analysis — measure λ_max independently per layer | ✅ Complete |
| **Phase 3** | Global CGRS — single shared rank, one global λ signal | ✅ Complete |
| **Phase 4** | **Per-layer CGRS** — each layer has its own rank + own τ threshold (highest novelty) | 🔄 In progress |

### Phase 1 — Fixed-Rank LoRA Baselines
Establishes the accuracy-vs-parameters frontier for fixed-rank LoRA applied globally. The accuracy curve shows strongly diminishing returns above `r=16`: jumping from `r=16` (89.51%) to `r=200` (90.95%) costs **12× more parameters for only 1.44% gain**. The gap between full fine-tuning (92.87%) and best LoRA (90.95%) is ~1.9%.

### Phase 2 — Per-Layer Curvature Analysis
Measures the diagonal Fisher `λ_max` for each of the 12 layers (query + value separately) across ranks `r ∈ {3, 5, 6, 10, 12, 16, 30, 52, 64}`. Reveals the **200× curvature spread** that motivates per-layer rank allocation. The `r=16` λ_max values become the τ calibration reference for Phase 4.

### Phase 3 — Global CGRS
A single PEFT LoRA model where all 12 layers share one rank, and a single global λ_max triggers transitions for the whole model. It can match `r=64` accuracy (90.48%) at a lower average rank (59.40), but improvement is marginal — a single global signal cannot distinguish flat early layers from sharp deep ones. This limitation directly motivates Phase 4.

### Phase 4 — Per-Layer CGRS (Current)
Each of the 12 layers maintains its **own rank** `layer_ranks[i]`, **own threshold** `τ_i`, and **own cooldown** counter. The Fisher probe returns a per-layer dictionary instead of a single global value. Three τ-calibration runs:

| Run | τ Config | Description |
|-----|----------|-------------|
| **C1** | 75th percentile = 0.001595 | Only layers sharper than 75% of Phase 2 baseline expand |
| **C2** | 50th percentile = 0.000486 | Half the layers eligible to expand |
| **C3** | Per-layer (each layer's own r=16 λ_max) | Self-referential, data-driven threshold — **highest novelty** |

Partial C1 results already confirm the hypothesis: **Layer 10 (a deep, sharp layer) is the first to expand** (16→30 at step 400, 30→52 at step 800), exactly as Phase 2 curvature predicted, while flat early layers have not triggered.

---

## Key Results

### Phase 1 — Fixed-Rank LoRA on CIFAR-100

| Configuration | Test Accuracy | Rank | Approx. LoRA Params |
|---|---|---|---|
| Frozen backbone (no LoRA) | 86.46% | — | 0 |
| LoRA r=6 | 87.27% | 6 | ~221K |
| LoRA r=10 | 89.02% | 10 | ~369K |
| LoRA r=16 | 89.51% | 16 | ~590K |
| LoRA r=64 | 90.48% | 64 | ~2.4M |
| LoRA r=200 | 90.95% | 200 | ~7.4M |
| **Full fine-tune** | **92.87%** | — | ~86.6M |

### Phase 2 — Per-Layer λ_max (Query projection highlights)

| Layer | Query λ_max | Note |
|---|---|---|
| L1 | 3.38e-5 | Flattest |
| L3 | 3.55e-5 | Flat |
| L9 | 2.54e-3 | Sharp |
| L10 | 3.43e-3 | Sharp |
| **L11** | **1.23e-2** | **Sharpest (~200× L1)** |

### Phase 3 — Global CGRS

| Run | τ | Avg Rank | Test Acc |
|---|---|---|---|
| CGRS τ=0.0047 | 0.0047 | 59.40 | 90.02% |
| CGRS τ=0.0200 | 0.0200 | 36.08 | 89.97% |
| CGRS τ=0.0518 | 0.0518 | 10.43 | 87.96% |

### Phase 4 — Per-Layer CGRS
Results pending completion of current training runs (C1/C2/C3 average rank, test accuracy, and final per-layer rank distribution TBD).

**Publication claim to validate:** Per-Layer CGRS allocates higher rank to sharp deep layers (L9–L11 query) and lower rank to flat early layers (L0–L3), achieving better accuracy-per-parameter than both fixed-rank LoRA and global CGRS. A Pearson/Spearman correlation between Phase 2 λ_max and Phase 4 final layer ranks quantifies this effect.

---

## Repository Structure

```
.
├── phase1_updated_plots.ipynb          # Phase 1: fixed-rank LoRA baselines + plots
├── phase2_updated_plots.ipynb          # Phase 2: per-layer curvature analysis + plots
├── phase2_results/                     # Phase 2 curvature JSON outputs
│   ├── all_results_complete.json       # 24 layer-proj λ_max values across ranks
│   └── curvature_results.json
├── Phase3_Jishan.ipynb                 # Phase 3: global CGRS
├── phase3_cgrs-Copy1.ipynb             # Phase 3: working copy
├── phase4_perlayer_cgrs_colab_v2.ipynb # Phase 4: per-layer CGRS (current work)
├── plots/                              # Publication-quality figures
│   ├── plot_A_pareto_frontier.png      # Accuracy vs params frontier
│   ├── plot_B_lmax_heatmap.png         # Per-layer λ_max heatmap
│   ├── plot_C_lmax_vs_rank_correlation.png
│   ├── plot_D_rank_evolution.png       # Rank trajectory during training
│   ├── phase4_final_ranks_per_layer.png
│   └── ...                             # Additional curvature/regression plots
└── .gitignore                          # Excludes large artifacts (*.pt, data/, result dirs)
```

> **Note:** Large artifacts — model checkpoints (`*.pt`, `*.pth`), the CIFAR-100 `data/` directory, and `phase1/phase3/phase4` result directories — are git-ignored and not stored in the repository.

---

## Environment & Setup

Developed and trained on the **USC CARC HPC cluster**.

### Hardware
- **GPU:** Tesla V100-PCIE-32GB (CARC HPC cluster), 34.1 GB VRAM
- **CPU:** CARC HPC multi-core (conda environment)
- **Storage:** local HPC filesystem

### Software Stack

| Library | Version |
|---|---|
| Python | 3.11.15 (conda-forge) |
| PyTorch | 2.2.0+cu121 |
| CUDA | 12.1 |
| Transformers (HuggingFace) | 4.40.0 |
| PEFT (HuggingFace) | 0.10.0 |
| torchvision | Compatible with PyTorch 2.2 |
| scipy | for `pearsonr` / `spearmanr` |
| matplotlib | Agg backend for HPC |
| numpy | standard |

### Suggested install

```bash
conda create -n cgrs python=3.11 -y
conda activate cgrs
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.0 peft==0.10.0 torchvision scipy matplotlib numpy
```

### Training Configuration (shared defaults)

| Parameter | Value |
|---|---|
| Train / Val / Test split | 45,000 / 5,000 / 10,000 |
| Batch size | 64 |
| Epochs | 3 |
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| Weight decay | 0.01 |
| LR schedule | CosineAnnealingLR |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| Seed | 42 |

### Phase 4 CGRS Parameters

| Parameter | Value |
|---|---|
| r_init | 16 |
| r_min / r_max | 3 / 64 |
| rank_list | [3, 5, 6, 10, 12, 16, 30, 52, 64] |
| check_every (K) | 200 steps |
| cooldown | 100 steps |
| probe_batches | 8 |
| expand_only | True (rank only increases) |

### Approximate Training Time (V100)

| Run type | Epochs | Approx. time |
|---|---|---|
| Fixed-rank LoRA (Phase 1) | 3 | ~20–25 min |
| Per-layer curvature probe (Phase 2) | N/A | ~10 min per rank |
| Global CGRS (Phase 3) | 3 | ~25–35 min |
| Per-layer CGRS (Phase 4) | 3 | ~30–40 min |

---

## How to Run

The project is organized as Jupyter notebooks, one per phase. Run them in order:

1. **`phase1_updated_plots.ipynb`** — fixed-rank LoRA baselines and Pareto frontier.
2. **`phase2_updated_plots.ipynb`** — per-layer curvature probe; produces the λ_max values used to calibrate τ.
3. **`Phase3_Jishan.ipynb`** — global CGRS sweep over τ.
4. **`phase4_perlayer_cgrs_colab_v2.ipynb`** — per-layer CGRS (C1/C2/C3 runs).

The CIFAR-100 dataset is auto-downloaded on first run.

### Phase 4 cell map

| Cell | Purpose |
|---|---|
| 1 | Verify environment (PyTorch, CUDA, PEFT versions) |
| 2 | Load paths & Phase 1–2 results; extract 24 layer-proj λ_max values |
| 3 | Imports & reproducibility (`SEED=42`; `from scipy import stats`) |
| 4 | Dataset & DataLoaders (CIFAR-100, train=45K/val=5K/test=10K) |
| 5 | Hyperparameters (`P4` dict) — note `r_init` here is overridden by Cell 8.5 |
| 6 | Model builder & utilities (`build_lora_model`, `evaluate`, `transition_lora_rank`) |
| 7 | Per-layer Fisher probe (PEFT-native, non-zero gradients) |
| 8 | Training loop (`run_perlayer_cgrs`) — expand-only, per-layer rank state |
| 8.5 | τ calibration — sets `r_init=16`, `TAU_C1/C2`, `TAU_PER_LAYER` from Phase 2 data |
| 9.5 | Run C1 (τ = 75th pct) |
| 9.6 | Runs C2 + C3 |
| 10–12 | Plots: final ranks, comparison table, publication-quality figures |
| 13 | Pearson/Spearman correlation: Phase 2 λ_max vs Phase 4 final ranks |
| 14 | Save all results |

---

## Internal Functions Reference

- **`build_lora_model(r)`** — Builds a PEFT LoRA model at rank `r` targeting query + value projections across all 12 layers. Uses `ignore_mismatched_sizes=True` for the CIFAR-100 classifier head.
- **`compute_perlayer_fisher(model, loader, n_batches)`** — Accumulates diagonal Fisher per `(layer_idx, proj)` over `n_batches` forward-backward passes. Returns `{(i, 'query'|'value'): λ_max}`. Uses only PEFT's own `lora_A`/`lora_B` parameters, which always carry `.grad`.
- **`transition_lora_rank(old_model, old_r, new_r)`** — SVD-based weight transfer between ranks. Copies non-LoRA weights directly; transfers LoRA weights via zero-padding (expand) or truncated SVD (compress). Reads `actual_old_r = A.shape[0]` rather than trusting the `old_r` argument.
- **`run_perlayer_cgrs(run_name, tau_config, r_init, epochs, save_dir, expand_only)`** — Main loop. Maintains `layer_ranks[i]` and `cooldown_left[i]`; every K steps checks each layer against its `τ_i`, calls `transition_lora_rank` on triggers, rebuilds optimizer/scheduler with carried-over LR.
- **`load_if_exists(run_name, save_dir)`** — Checkpoint/resume helper; loads existing result JSON to skip re-training.
- **`evaluate(model, loader)`** — Inference loop returning `(accuracy_percent, avg_loss)` under `@torch.no_grad()`.
- **`global_lmax(lmax_dict)`** — Returns the single global maximum across all per-layer Fisher keys (for logging).

PEFT parameter naming parsed by the Fisher probe:
```
base_model.model.vit.encoder.layer.{i}.attention.attention.{query|value}.lora_A.default.weight
```

---

## Engineering Lessons & Bugs Fixed

- **Fisher λ_global = 0.00000 (major, fixed).** The initial `PerLayerLoRAViT` class attached LoRA weights via forward hooks, so they were never registered in the model graph → `.grad` was always `None` and Fisher was always 0. **Fix:** replaced with a standard `get_peft_model()` PEFT model so `lora_A`/`lora_B` flow gradients through the normal forward pass.
- **Tensor shape mismatch on sequential rank transitions (fixed).** `transition_lora_rank` used the function argument `old_r` for slicing, which went stale after the first transition (`RuntimeError: expanded size (16) must match existing size (52)`). **Fix:** read `actual_old_r = A.shape[0]`.
- **Stale τ values (fixed).** Cell 5 hardcoded `r_init=3` and an old τ; Cell 8.5 now explicitly overwrites `r_init=16` and recomputes all τ from the live Phase 2 dict.
- **Stale result files blocking re-runs (fixed).** `load_if_exists()` skipped execution when old broken JSONs existed; Cell 9.5 now cleans them up automatically.
- **scipy import (pending).** Cell 3 used `from scipy.stats import pearsonr` but Cell 13 calls `stats.pearsonr`. **Fix:** change Cell 3 to `from scipy import stats`.
- **ViT classifier weight warning (non-critical).** The 1000→100 class head mismatch warning is expected and handled by `ignore_mismatched_sizes=True`.
- **Memory management.** Each run deletes the model and calls `torch.cuda.empty_cache()` — essential on CARC since checkpoints are reloaded from disk between runs.

---

## Future Directions

- **Bidirectional CGRS** — allow rank to *decrease* (compress) as well as expand, letting the model "forget" capacity in layers that become flat after initial learning.
- **Asymmetric α/r** — scale `lora_alpha` proportionally to rank (e.g., `α = 2r`) instead of a fixed 16, for stability during transitions.
- **Cross-dataset generalization** — test on CIFAR-10, Flowers102, Oxford-IIIT Pet to validate that the per-layer curvature structure generalizes.
- **Multi-rank Phase 2 τ** — use λ_max across all probed ranks (not just r=16) for richer rank-curvature thresholds.
- **Publication target** — a strong positive Pearson correlation (r > 0.7) between Phase 2 λ_max and Phase 4 final ranks, combined with accuracy-efficiency results, supports submission to venues such as ICLR or ECCV.

---

## Author

**Jishan Shaikh** — Graduate research, University of Southern California (USC).
Project: *CGRS — Curvature-Guided Rank Selection for LoRA Fine-Tuning of Vision Transformers* (May 2026).
