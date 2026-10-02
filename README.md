# Curvature-Guided Rank Scheduling (CGRS) for LoRA Fine-Tuning of Vision Transformers

Code, logs, and result files for **CGRS**, a family of schedulers that change LoRA rank *during* fine-tuning using a diagonal empirical-Fisher curvature signal. Experiments use an ImageNet-pretrained ViT-Base/16 on CIFAR-100, SVHN, and Oxford Flowers-102.

> This repository accompanies a paper under double-blind review. It was originally named "...Rank-Selection..."; the method is now called **Rank Scheduling**.

**Status: pilot study.** Every configuration is a single seed (42). Differences of 0.5–1.5 points may be within run-to-run noise. See [Limitations and implementation notes](#limitations-and-implementation-notes).

---

## Contents
1. [Idea](#idea)
3. [Headline results](#headline-results)
4. [Method in detail](#method-in-detail)
5. [Experimental setup](#experimental-setup)
6. [Full results](#full-results)
7. [Repository structure](#repository-structure)
8. [Environment and running](#environment-and-running)
9. [Result file formats](#result-file-formats)
10. [Limitations and implementation notes](#limitations-and-implementation-notes)

---

## Idea

LoRA fixes its rank `r` before training and shares it across layers. CGRS treats rank as a training-time variable. It probes the diagonal empirical Fisher of the LoRA parameters (squared mini-batch gradients), summarizes it globally or per transformer block, and compares it with calibrated thresholds:

- **Global CGRS** uses one shared rank that can grow *and* shrink along a discrete ladder.
- **Layer-wise CGRS** assigns one rank to each of the 12 blocks (query and value share it) and can only *grow*.
- Every rank change carries over the learned low-rank update.

![Rank trajectories](figures/fig3_rank_trajectory.png)
![Per-layer final ranks](figures/fig4_module_heatmap.png)

---

## Headline results

Test accuracy (%), ViT-Base/16, single seed.

| Dataset | Global CGRS (best) | Fixed LoRA, same final rank | Δ | Best AdaLoRA | Full FT | Frozen backbone (trained head) |
|---|---|---|---|---|---|---|
| CIFAR-100 | **91.16** (τ=75th pct, final r=64) | 90.46 | +0.70 | 90.06 | 92.55 | 86.34 |
| SVHN | **97.45** (τ=75th pct, final r=52) | 96.98 | +0.47 | 95.91 | 97.77 | 55.06 |
| Flowers-102 | 98.52 (τ=25th pct, final r=64) | 98.73 | −0.21 | **98.78** | 99.01 | 99.20 |

- CIFAR-100 and SVHN: global CGRS beats fixed LoRA at the same final rank and the best AdaLoRA run.
- Flowers-102 reverses the pattern. AdaLoRA wins, and some layer-wise schedules collapse to the maximum rank (a threshold-calibration failure).
- "Best" CGRS and AdaLoRA runs were chosen by test accuracy among 3 configurations each.

---

## Method in detail

### 1. LoRA parameterization
For a frozen projection `W0 ∈ R^{d_out×d_in}`:

`h = W0·x + (α/r)·B·A·x`, with `A ∈ R^{r×d_in}`, `B ∈ R^{d_out×r}`.

LoRA is applied to the **query and value** projections of all 12 ViT blocks (24 adapters, `d_in = d_out = 768`). A uniform rank uses `24·(768+768)·r = 36,864·r` adapter parameters. `α = 16`, dropout 0.1, bias none.

### 2. Diagonal empirical-Fisher curvature score
For probe mini-batches `b` with gradient `g_b = ∇θ ℓ_b`, the estimator is `f̂ = mean_b (g_b ⊙ g_b)`. The scheduler uses:

- **Module/block score** `λ_m = max_i f̂_i` over the coordinates of the adapter(s).
- **Global score** `λ = max` over all trainable LoRA coordinates.
- The trace `Σ f̂_i` is recorded for diagnostics only.

Details:
- Gradients are mini-batch-mean-loss gradients, not per-sample. The model is in `eval()` mode while probing.
- Static profiling uses batches of size 4. In-training probes use training batches of size 16.

### 3. Threshold calibration (Phases 2 and 4)
- **Static (Phase 2):** Train fixed-rank LoRA at all 14 ranks. For each checkpoint, compute `λ` globally and per `(block, q/v)` over 64 probe batches of size 4 (256 samples). The 24 per-module scores at `r = 16` give the **25th / 50th / 75th percentiles**, used as the *aggressive / moderate / conservative* thresholds `τ`.
- **Live (Phase 4, PL-C3-Live):** Train a fresh `r = 30` model for 800 warm-up steps (same seed), measure per-module `λ` over 16 probe batches, multiply by a ×4 safety factor, and later multiply by 0.5 (net ×2). A block's threshold is the larger of its query and value values.

### 4. Global scheduler
- Rank ladder `R = {3, 5, 6, 10, 12, 16, 30, 52, 64}`, initial `r = 16`.
- Probes every **200** steps, after a **15% grace period**, and no earlier than **400 steps** after the previous change. Each probe uses 8 batches.
- If `λ > τ`, move up `max(1, floor(λ/τ))` rungs (capped at the top). If `λ < 0.3τ`, move down one rung. Otherwise keep the rank.
- After a resize the learning rate is multiplied by `r_new / r_old`, the AdamW optimizer is rebuilt (state reset), and a new cosine schedule runs over the remaining steps.

### 5. Layer-wise schedulers
All start every block at `r = 30`, bounded to `[3, 64]`, with probes every 200 steps (16 batches) and growth only. In the CIFAR-100 implementation there is no grace period. After a block changes, a 200-step counter is set. It is decremented at each check, so the effective gap between changes of one block is 400 steps. Resizing a block resets only the Adam state of its two adapters. The learning-rate schedule is not changed.

| Policy | Rule |
|---|---|
| **PL-C1** | Block score (max of q/v `λ`) is compared with one shared static threshold: the median (50th pct) of Phase 2 `r=16` scores. |
| **PL-C3-Live** | Each block gets its own threshold from the live calibration. |
| **Ordinal-K4 / K5** | At each check, rank the 12 blocks by `λ` and grow the top 4 (or 5). Jump size is `floor(score / median score)` rungs (minimum 1). Blocks 4–7 have a hand-chosen minimum rank of 16 (note: starting rank is 30, so this floor never binds in this configuration). |

For threshold policies, the jump size is `max(1, floor(λ_block / τ_block))` rungs, capped at the top rung (rank 64).

### 6. Rank transfer
- **Shrink:** form `ΔW = B·A`, take a truncated SVD, and set `B' = U_r·√Σ_r`, `A' = √Σ_r·V_rᵀ`. This is the best rank-`r'` approximation in Frobenius norm.
- **Grow:** the existing `A` and `B` are copied into the larger matrices and the added rows/columns are **zero-padded**.
- The scaling `α/r` is updated to the new rank, so the scaled update `(α/r)·B·A` is not exactly preserved across a resize.

---

## Experimental setup

| | |
|---|---|
| Backbone | `google/vit-base-patch16-224` (Hugging Face) |
| Preprocessing | Resize to 224×224, ToTensor, normalize with the ViT image processor mean/std. **No augmentation.** |
| Optimizer | AdamW, weight decay 0.01, batch size 16, cosine decay (stepped per iteration), gradient clipping at 1.0 |
| Learning rates | Full FT 5e-5, frozen backbone + head 1e-3, LoRA-family (fixed, CGRS, AdaLoRA) 5e-4 |
| LoRA | targets `query`, `value`; α = 16; dropout 0.1; bias none |
| Seed | 42 (Python, NumPy, PyTorch, CUDA). `cudnn.benchmark = True`, so runs are not bit-reproducible. |
| Hardware | NVIDIA A100 and V100 GPUs, used interchangeably across runs (no single GPU throughout) |

| Dataset | Train / Val / Test | Epochs | Steps |
|---|---|---|---|
| CIFAR-100 | 45,000 / 5,000 / 10,000 (seeded random split of the train set) | 3 | 8,439 |
| SVHN | 68,000 / 5,257 / 26,032 (`train` split only; no `extra`) | 3 | 12,750 |
| Oxford Flowers-102 | 1,840 / 200 / 6,149 (official train+val concatenated, then seeded re-split; test untouched) | 20 | 2,300 |

**Flowers-102:** with only 2,300 steps, check interval and cooldown are given as fractions of training: 2.4% (~55 steps) and 4.7% (~108 steps). The 15% grace period still applies.

**Baselines:**
- Full fine-tuning.
- Frozen backbone with a trained classifier head.
- Fixed LoRA at `r ∈ {3, 5, 6, 10, 12, 16, 30, 52, 64, 80, 100, 128, 150, 200}`.
- AdaLoRA (Hugging Face PEFT 0.10.0) with target ranks 16, 32, 64, initial rank `int(1.5 × target)`, `tinit` = `tfinal` = 15% of total steps, `deltaT` = 2.4% of total steps, same optimizer, LR, and schedule as the other LoRA runs. `update_and_allocate(step)` runs after every optimizer step.

**Experimental phases** (in `CGRS_CIFAR100_Rebuild_AllPhases.py`):
1. Baselines and the 14-rank fixed LoRA sweep (checkpoints saved).
2. Curvature profiling of every Phase 1 checkpoint, giving percentile thresholds.
3. Global CGRS at three thresholds (aggressive, moderate, conservative).
4. Layer-wise CGRS: PL-C1, PL-C3-Live, Ordinal-K4, Ordinal-K5, plus rank–curvature correlation analysis (Pearson and Spearman, against both static and live curvature).

---

## Full results

### Fixed-rank LoRA sweep (test acc. %)

| Rank | Params (M) | CIFAR-100 | SVHN | Flowers-102 |
|---|---|---|---|---|
| 3 | 0.111 | 85.07 | 96.43 | 95.56 |
| 5 | 0.184 | 87.13 | 96.90 | 96.80 |
| 6 | 0.221 | 87.50 | 96.80 | 96.57 |
| 10 | 0.369 | 89.36 | 96.57 | 98.00 |
| 12 | 0.442 | 89.15 | 96.75 | 98.41 |
| 16 | 0.590 | 89.74 | 96.58 | 98.15 |
| 30 | 1.106 | 90.03 | 96.48 | 98.03 |
| 52 | 1.917 | 90.54 | 96.98 | 98.24 |
| 64 | 2.359 | 90.46 | 96.83 | 98.73 |
| 80 | 2.949 | 90.54 | 96.75 | 98.70 |
| 100 | 3.686 | 90.40 | 96.63 | 98.86 |
| 128 | 4.719 | 91.07 | 96.98 | 98.89 |
| 150 | 5.530 | 90.70 | 96.88 | 98.75 |
| 200 | 7.373 | 90.62 | 96.80 | 98.54 |

### Conventional baselines

| Dataset | Full FT (params; acc.) | Frozen backbone (params; acc.) |
|---|---|---|
| CIFAR-100 | 85,875,556; 92.55 | 76,900; 86.34 |
| SVHN | 85,806,346; 97.77 | 7,690; 55.06 |
| Flowers-102 | 85,877,094; 99.01 | 78,438; 99.20 |

### All global CGRS runs

| Dataset | Threshold | Final r | Rank changes | Params (M) | Acc. |
|---|---|---|---|---|---|
| CIFAR-100 | 25th | 64 | 1 | 2.359 | 90.72 |
| CIFAR-100 | 50th | 64 | 6 | 2.359 | 90.62 |
| CIFAR-100 | 75th | 64 | 8 | 2.359 | **91.16** |
| SVHN | 25th | 52 | – | 1.917 | 97.03 |
| SVHN | 50th | 64 | – | 2.359 | 97.21 |
| SVHN | 75th | 52 | – | 1.917 | **97.45** |
| Flowers-102 | 25th | 64 | – | 2.359 | **98.52** |
| Flowers-102 | 50th | 64 | – | 2.359 | 98.28 |
| Flowers-102 | 75th | 30 | – | 1.106 | 97.01 |

On Flowers-102, curvature values are about 1e-5 to 1e-4 and the global scheduler oscillates, for example 16→64→52→30→16→64 in one conservative run.

### Layer-wise CGRS

| Dataset | Scheduler | Avg. rank | Params (M) | Acc. | Final ranks |
|---|---|---|---|---|---|
| CIFAR-100 | PL-C1 | 45.83 | 1.690 | 89.13 | heterogeneous |
| CIFAR-100 | PL-C3-Live | 37.33 | 1.376 | **90.18** | heterogeneous |
| CIFAR-100 | Ordinal-K4 | 61.00 | 2.249 | 88.68 | heterogeneous |
| CIFAR-100 | Ordinal-K5 | 63.00 | 2.322 | 88.66 | heterogeneous |
| SVHN | PL-C1 | 39.33 | 1.450 | 96.55 | heterogeneous |
| SVHN | PL-C3-Live | 62.00 | 2.286 | **96.75** | heterogeneous |
| SVHN | Ordinal-K4 | 60.00 | 2.212 | 96.72 | heterogeneous |
| SVHN | Ordinal-K5 | 64.00 | 2.359 | 96.72 | uniform |
| Flowers-102 | PL-C1 | 63.00 | 2.322 | 97.92 | heterogeneous |
| Flowers-102 | PL-C3-Live | 64.00 | 2.359 | **98.26** | uniform |
| Flowers-102 | Ordinal-K4 | 63.00 | 2.322 | 98.23 | heterogeneous |
| Flowers-102 | Ordinal-K5 | 63.00 | 2.322 | 98.11 | heterogeneous |

- PL-C3-Live on CIFAR-100 uses 42% fewer final adapter parameters than fixed `r = 64`, at 0.28 points lower accuracy.
- SVHN accuracy is nearly flat across ranks (96.43–96.98 for r = 3–200), so low allocations stay competitive.
- On Flowers-102, layer-wise methods converge to ranks of 63–64 (PL-C3-Live is uniformly 64). That is an allocation failure, not evidence every block needs maximum rank.

### AdaLoRA (PEFT)

| Dataset | Target r | Stored params (M) | Acc. |
|---|---|---|---|
| CIFAR-100 | 16 / 32 / 64 | 0.885 / 1.771 / 3.541 | 89.25 / 89.71 / 90.06 |
| SVHN | 16 / 32 / 64 | 0.885 / 1.771 / 3.541 | 95.91 / 95.74 / 95.86 |
| Flowers-102 | 16 / 32 / 64 | 0.885 / 1.771 / 3.541 | 98.55 / 98.78 / 98.60 |

AdaLoRA's stored parameters reflect its over-parameterized initialization (init rank = 1.5 × target, including its per-triplet singular values), so rank-matched comparisons are not parameter-matched.

### Does curvature drive allocation?
Final block ranks were correlated with static (`r=16`) and live curvature (Pearson and Spearman, 24 adapter points). The clearest case is SVHN PL-C1 vs. static curvature (Pearson r = 0.4325, p = 0.0348). Other non-uniform runs are mostly positive but not significant (p ≥ 0.05). Because q/v share a rank, points are paired, so these are descriptive. Uniform runs make correlation undefined.

---

## Repository structure

```text
.
├── CGRS_Buildup/
│   └── rebuild_all_phases/
│       ├── CGRS_CIFAR100_Rebuild_AllPhases.py   # Full CGRS pipeline: Phases 1-4 + analysis (CIFAR-100)
│       ├── test_inplace_resize.py               # Checks in-place resize leaves other layers' optimizer state untouched
│       └── test_rank_pattern.py                 # Checks per-layer rank_pattern construction
├── AdaLORA_Comparison/
│   ├── CGRS_AdaLoRA_Comparison.py               # AdaLoRA baselines (--dataset cifar100|svhn|flowers102)
│   └── adalora_{cifar100,flowers102,svhn}.log
├── CIFAR100_Rebuild/                            # CIFAR-100 results reported in the paper (3-epoch run)
│   ├── phase1_results/ ... phase4_results/
│   ├── adalora_results/
│   └── all_results_complete.json
├── SVHN_Rebuild/                                # same structure
├── Flowers102_Rebuild/                          # same structure
├── figures/                                     # fig3_rank_trajectory.png, fig4_module_heatmap.png
├── generate_paper_figures.py                    # builds both figures from the all_results_complete.json files
└── README.md
```


> Note: the pipeline script in this repo is the CIFAR-100 version. SVHN and Flowers-102 CGRS runs used the same pipeline with the dataset-specific settings in the tables above. [Add those scripts here or describe the differences.]

Large artifacts (datasets, `*.pt` checkpoints) are excluded via `.gitignore`.

---

## Environment and running

Tested configuration:

| Component | Version |
|---|---|
| Python | 3.11.15 |
| PyTorch / torchvision | 2.2.0+cu121 / 0.17.0+cu121 (CUDA 12.1) |
| Transformers | 4.40.0 |
| PEFT | 0.10.0 |
| Accelerate / Datasets | 1.13.0 / 4.8.5 |
| NumPy / SciPy / pandas / Matplotlib | 1.26.4 / 1.17.1 / 3.0.3 / 3.10.9 |
| GPU | NVIDIA A100 and V100 (mixed across runs) |

```bash
python -m venv cgrs_env && source cgrs_env/bin/activate
pip install torch==2.2.0 torchvision==0.17.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install transformers==4.40.0 peft==0.10.0 accelerate==1.13.0 datasets==4.8.5 \
            numpy==1.26.4 scipy==1.17.1 pandas matplotlib
```

**Run the CGRS pipeline** (CIFAR-100):

```bash
python CGRS_Buildup/rebuild_all_phases/CGRS_CIFAR100_Rebuild_AllPhases.py
```

- Output paths are set at the top of the script: `PROJECT_BASE = ~/CGRS_Project/CIFAR100_Rebuild`. Change this for your machine.
- CIFAR-100 downloads automatically into `PROJECT_BASE/data`.
- The script runs Phases 1→4 in order and **skips** anything already saved: Phase 1 results and checkpoints, Phase 2 curvature, and each run's `*_result.json`.
- Layer-wise runs checkpoint after every epoch (`*_progress.json/.pt`) and resume automatically. Global runs cannot resume mid-run.
- Phase 1 stores 14 LoRA checkpoints (`lora_r*_full.pt`). They are large, so they are gitignored.

**AdaLoRA baselines** (3 target ranks per dataset, results in `<dataset>_Rebuild/adalora_results/`):

```bash
python AdaLORA_Comparison/CGRS_AdaLoRA_Comparison.py --dataset cifar100   # or svhn, flowers102
```

The `project_base` paths for the three datasets are set in `DATASET_CONFIG` at the top of the script (`~/CGRS_Project/<Dataset>_Rebuild`).

**Figures** (run from the repository root):

```bash
python generate_paper_figures.py
```

- Figure 3 plots the global-CGRS rank trajectory (rank vs. % of training, starting at `r = 16`) for the three thresholds on each dataset, with the 15% grace period shaded. It reads `phase3 → rank_changes`.
- Figure 4 is a heatmap of final per-block ranks for three layer-wise runs: SVHN PL-C1, CIFAR-100 PL-C3-Live, and Flowers-102 PL-C3-Live. It reads `phase4 → final_ranks`.
- No numbers are hardcoded; both figures are built from each dataset's `all_results_complete.json`.

**Sanity tests:** `python CGRS_Buildup/rebuild_all_phases/test_inplace_resize.py` and `test_rank_pattern.py`

### Reproducing the paper's numbers

Every number in the paper's tables comes from the committed files: fixed and baseline results from `phase1_results/`, thresholds from `phase2_results/`, global CGRS from `phase3_results/`, layer-wise from `phase4_results/`, and AdaLoRA from `adalora_results/`. Each dataset's `all_results_complete.json` combines the CGRS phases. Reruns on different GPUs (A100 vs. V100) may differ slightly.

---

## Result file formats

| File | Contents |
|---|---|
| `phase1_results/config.json` | The `CONFIG` dictionary used |
| `phase1_results/results.json` | Test acc., loss, and trainable params for Full FT, Frozen, and all 14 fixed ranks |
| `phase2_results/curvature_results.json` | Per-rank `lambda_max`, `trace`, and per-(block, q/v) `λ_max` |
| `phase3_results/CGRS_Global_tau_{aggressive,moderate,conservative}_result.json` | `tau`, `final_rank`, `test_acc`, `total_lora_params`, `n_rank_changes`, and the `rank_changes` log (step, old/new rank, λ) |
| `phase4_results/{PL_C1,PL_C3_Live,OrdinalK4_Fixed,OrdinalK5}_result.json` | See schema below |
| `phase4_results/live_calibration.json` | Live per-module `λ` and percentiles |
| `adalora_results/AdaLoRA_target_r{16,32,64}_result.json`, `adalora_all_results.json` | `init_r`, `target_r`, `test_acc`, `test_loss`, `total_lora_params` |
| `all_results_complete.json` | `phase1`, `phase2_layer_lmax_r16`, `phase2_percentiles`, `phase3`, `phase4`, `phase4_live_calibration` combined |

**Layer-wise result schema:**

```json
{
  "run_name": "...",
  "tau": "per-layer | <float>",
  "r_init": 30,
  "test_acc": 0.0,
  "test_loss": 0.0,
  "avg_rank": 0.0,
  "final_ranks": {"0": 0, "1": 0, "...": 0, "11": 0},
  "total_lora_params": 0,
  "n_rank_changes": 0,
  "rank_changes": [
    {"step": 0, "layer": 0, "old_r": 0, "new_r": 0, "lambda": 0.0, "tau": 0.0}
  ]
}
```

Ordinal runs also store `top_k`, `protected_layers`, and `protected_min_rank`, and log a `reason` (`topk` or `floor`) per change instead of `lambda`/`tau`. Adapter parameters satisfy `total_lora_params = 3072 × Σ final_ranks`.

---

## Limitations and implementation notes

**Statistical.**
- Single seed per configuration. The study uses one backbone and classification only.
- "Best" CGRS and AdaLoRA configurations were picked by test accuracy.
- GPU type (A100 vs. V100) varied between runs and `cudnn.benchmark=True`, so exact numbers may shift on rerun.
- Final adapter counts do not measure peak memory, wall-clock time, or the overhead of probes and SVD.
- No direct comparison with the concurrent LAARA, GRIT, or CG-LoRA methods.

**Implementation notes (read before interpreting the gains).**
- **Classifier head.** In the LoRA-family runs (fixed LoRA, CGRS, AdaLoRA) the classification head is created with random initialization and is *not* trained: only adapter parameters have `requires_grad=True`. Only the "Frozen backbone" baseline trains its head. Adapter parameter counts therefore cover the adapters only.
- **Rank growth.** Growth zero-pads both LoRA factors. Since the gradient of each factor is proportional to the other, newly added directions receive zero gradient, so growth mainly changes the `α/r` scaling (and, for the global scheduler, the learning rate) instead of adding trainable capacity. The observed gains over fixed LoRA at equal final rank should be read as a schedule/scaling effect. A growth rule that initializes new `A` rows randomly (as in standard LoRA) is untested.
- **Train/eval mode.** Fisher probes and evaluation put the model in `eval()` mode and training mode is not restored afterward. In the CGRS runs, LoRA dropout (0.1) is therefore inactive after the first probe or evaluation. In the AdaLoRA runs it is inactive after the first epoch. Fixed-rank baselines train with dropout throughout.
- **Grace period.** The global scheduler waits 15% of training before its first change. In the CIFAR-100 layer-wise runs, probing and rank changes begin at step 200 with no grace period.
- **AdaLoRA regularizer.** The loss is computed outside the PEFT forward pass, so AdaLoRA's orthogonality regularizer is likely not applied.
- **Probe settings.** Static profiling used 64 probe batches of size 4; in-training probes use size-16 batches. The global scheduler compares a maximum over all coordinates with percentiles of per-module maxima.
- **Randomness.** The random classifier head differs across runs because the RNG is seeded only once per script.

**Next steps:** multi-seed replication, trainable classifier head, random-initialized growth directions, restoring train mode after probes, shrinkable layer-wise allocation, threshold normalization or hysteresis, and measurement of training cost.
