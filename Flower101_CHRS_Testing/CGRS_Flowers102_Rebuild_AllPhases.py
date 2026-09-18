"""
CGRS on Oxford Flowers-102 -- Full Pipeline (Phases 1-4)
Mirrors the validated CIFAR-100 / SVHN pipelines exactly, with ONE
methodological change required for a fair cross-dataset comparison:

  Flowers102 has a tiny training pool (official train+val = 2,040 images),
  giving far fewer total training steps than CIFAR-100 (~8,400) or SVHN
  (~12,750) at the same batch size and epoch count. Using the same FIXED
  check_every/cooldown (200/400 steps) that worked for those datasets would
  mean the rank-adaptation mechanism barely fires here -- an apples-to-oranges
  comparison. So check_every, cooldown, and the grace period are expressed
  as FRACTIONS of total training steps, calibrated to reproduce the same
  *check density* (~40-60 adaptation checks per run) regardless of dataset
  size. This is the only structural change; every other hyperparameter
  (rank lists, LoRA alpha/dropout, r_init, threshold percentile methodology,
  epochs) is identical across all datasets for a controlled comparison.
"""
import os, json, time, random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from peft import LoraConfig, get_peft_model
from scipy.stats import pearsonr, spearmanr

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("WARNING: No GPU detected.")

# ============================================================================
# Cell 2 -- Config and Directories (Flowers-102)
# ============================================================================
CONFIG = {
    "model_name": "google/vit-base-patch16-224",
    "num_classes": 102,
    "internal_val_size": 200,     # carved out of the combined train+val pool
    "batch_size": 16,
    "epochs": 20,
    "weight_decay": 0.01,
    "lr_full": 5e-5, "lr_frozen": 1e-3, "lr_lora": 5e-4,
    "lora_alpha": 16, "lora_dropout": 0.1,
    "target_modules": ["query", "value"],
    "lora_ranks": [3, 5, 6, 10, 12, 16, 30, 52, 64, 80, 100, 128, 150, 200],
}

# Fair cross-dataset adaptation cadence: fractions of total steps, calibrated
# to match the check density used on CIFAR-100 (check_every=200 / ~8400
# total steps =~ 2.4%) and (cooldown=400 / ~8400 =~ 4.7%).
CHECK_FRACTION = 0.024
COOLDOWN_FRACTION = 0.047
GRACE_FRACTION = 0.15   # same as other datasets -- no rank changes before this point

PROJECT_BASE = os.path.expanduser("~/CGRS_Project/Flowers102_Rebuild")
PHASE1_DIR = os.path.join(PROJECT_BASE, "phase1_results")
PHASE2_DIR = os.path.join(PROJECT_BASE, "phase2_results")
PHASE3_DIR = os.path.join(PROJECT_BASE, "phase3_results")
PHASE4_DIR = os.path.join(PROJECT_BASE, "phase4_results")
DATA_DIR = os.path.join(PROJECT_BASE, "data")
for d in [PHASE1_DIR, PHASE2_DIR, PHASE3_DIR, PHASE4_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)
with open(os.path.join(PHASE1_DIR, "config.json"), "w") as f:
    json.dump({**CONFIG, "check_fraction": CHECK_FRACTION,
               "cooldown_fraction": COOLDOWN_FRACTION, "grace_fraction": GRACE_FRACTION}, f, indent=2)
print(f"PROJECT_BASE: {PROJECT_BASE}")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# ============================================================================
# Cell 3 -- Dataset and DataLoaders (Flowers-102)
# Official splits: train=1020, val=1020, test=6149. We combine train+val
# into one pool (2040 images), carve our own small internal val out of it,
# and hold the full official test split untouched for final evaluation --
# giving a train/val/test structure comparable to the other datasets.
# ============================================================================
processor = ViTImageProcessor.from_pretrained(CONFIG["model_name"])
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

print("Loading Flowers-102...")
official_train = datasets.Flowers102(root=DATA_DIR, split="train", download=True, transform=transform)
official_val = datasets.Flowers102(root=DATA_DIR, split="val", download=True, transform=transform)
test_dataset = datasets.Flowers102(root=DATA_DIR, split="test", download=True, transform=transform)

combined_pool = ConcatDataset([official_train, official_val])
pool_size = len(combined_pool)
val_size = CONFIG["internal_val_size"]
train_size = pool_size - val_size
train_dataset, val_dataset = random_split(
    combined_pool, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

NUM_WORKERS = 4
train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
probe_loader = DataLoader(train_dataset, batch_size=4, shuffle=False,
                           num_workers=0, pin_memory=False)

TOTAL_STEPS_PER_RUN = CONFIG["epochs"] * len(train_loader)
CHECK_EVERY = max(1, round(CHECK_FRACTION * TOTAL_STEPS_PER_RUN))
COOLDOWN = max(1, round(COOLDOWN_FRACTION * TOTAL_STEPS_PER_RUN))

print(f"Train: {len(train_dataset):,}  Val: {len(val_dataset):,}  Test: {len(test_dataset):,}")
print(f"Steps/epoch: {len(train_loader)}  Total steps/run: {TOTAL_STEPS_PER_RUN}")
print(f"Derived check_every={CHECK_EVERY}  cooldown={COOLDOWN}  "
      f"(~{TOTAL_STEPS_PER_RUN // CHECK_EVERY} checks per run)")

# ============================================================================
# Cell 4 -- Model Builders and Core Training Utilities
# ============================================================================
def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def build_full_model():
    model = ViTForImageClassification.from_pretrained(
        CONFIG["model_name"], num_labels=CONFIG["num_classes"], ignore_mismatched_sizes=True)
    return model.to(device)

def build_frozen_model():
    model = ViTForImageClassification.from_pretrained(
        CONFIG["model_name"], num_labels=CONFIG["num_classes"], ignore_mismatched_sizes=True)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    return model.to(device)

def build_lora_model(r):
    base = ViTForImageClassification.from_pretrained(
        CONFIG["model_name"], num_labels=CONFIG["num_classes"], ignore_mismatched_sizes=True)
    cfg = LoraConfig(r=r, lora_alpha=CONFIG["lora_alpha"], lora_dropout=CONFIG["lora_dropout"],
                      target_modules=CONFIG["target_modules"], bias="none")
    return get_peft_model(base, cfg).to(device)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(pixel_values=imgs)
        loss_sum += criterion(out.logits, labels).item()
        correct += (out.logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total, loss_sum / len(loader)

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_optimizer(model, lr, weight_decay):
    return optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=lr, weight_decay=weight_decay)

def get_scheduler(optimizer, total_steps):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

def train_fixed_model(model, lr, label, epochs=CONFIG["epochs"]):
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, lr, CONFIG["weight_decay"])
    scheduler = get_scheduler(optimizer, epochs * len(train_loader))
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(pixel_values=imgs)
            loss = criterion(out.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        val_acc, val_loss = evaluate(model, val_loader)
        print(f"  {label} Epoch {epoch}/{epochs} Train Loss {running_loss/len(train_loader):.4f} "
              f"Val Acc {val_acc:.2f}%")
    return model

def transition_lora_rank(old_model, old_r, new_r):
    old_params = {n: p.data.clone().cpu() for n, p in old_model.named_parameters()}
    new_model = build_lora_model(new_r).cpu()
    new_param_dict = dict(new_model.named_parameters())
    for name, old_data in old_params.items():
        if "lora_A" not in name and "lora_B" not in name and name in new_param_dict:
            new_param_dict[name].data.copy_(old_data)
    lora_bases = {}
    for name, data in old_params.items():
        if "lora_A" in name:
            base = name[:name.index("lora_A")]
            lora_bases.setdefault(base, {})["A"] = data
        elif "lora_B" in name:
            base = name[:name.index("lora_B")]
            lora_bases.setdefault(base, {})["B"] = data
    for base, pair in lora_bases.items():
        if "A" not in pair or "B" not in pair:
            continue
        A, B = pair["A"].float(), pair["B"].float()
        actual_old_r = A.shape[0]
        if new_r > actual_old_r:
            new_A = torch.zeros(new_r, A.shape[1]); new_A[:actual_old_r] = A
            new_B = torch.zeros(B.shape[0], new_r); new_B[:, :actual_old_r] = B
        else:
            W = B @ A
            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                sqrt_S = torch.sqrt(S[:new_r].clamp(min=0.0))
                new_B = U[:, :new_r] * sqrt_S
                new_A = Vh[:new_r] * sqrt_S.unsqueeze(1)
            except RuntimeError:
                new_A, new_B = A[:new_r], B[:, :new_r]
        for name, param in new_model.named_parameters():
            if name.startswith(base) and "lora_A" in name:
                param.data.copy_(new_A.to(param.dtype))
            elif name.startswith(base) and "lora_B" in name:
                param.data.copy_(new_B.to(param.dtype))
    return new_model.to(device)

print("Model builders and training utilities ready.")

# ============================================================================
# Cell 5 -- Fisher / Curvature Utilities
# ============================================================================
def compute_global_fisher(model, loader, n_batches=16):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    fisher, done = None, 0
    for imgs, labels in loader:
        if done >= n_batches:
            break
        imgs, labels = imgs.to(device), labels.to(device)
        model.zero_grad()
        loss = criterion(model(pixel_values=imgs).logits, labels)
        loss.backward()
        with torch.no_grad():
            grads = torch.cat([p.grad.detach().pow(2).flatten()
                                for p in model.parameters() if p.requires_grad and p.grad is not None])
        fisher = grads if fisher is None else fisher + grads
        done += 1
    fisher /= done
    return {"lambda_max": float(fisher.max().item()), "trace": float(fisher.sum().item()),
            "num_params": int(fisher.shape[0])}

def compute_perlayer_fisher(model, loader, n_batches=16):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    fisher_accum, batches_done = {}, 0
    for imgs, labels in loader:
        if batches_done >= n_batches:
            break
        imgs, labels = imgs.to(device), labels.to(device)
        model.zero_grad()
        loss = criterion(model(pixel_values=imgs).logits, labels)
        loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is None or not param.requires_grad:
                    continue
                if "lora_A" not in name and "lora_B" not in name:
                    continue
                parts = name.split(".")
                layer_idx, proj = None, None
                for j, p in enumerate(parts):
                    if p == "layer" and j + 1 < len(parts):
                        try:
                            layer_idx = int(parts[j + 1])
                        except ValueError:
                            pass
                    if p in ("query", "value"):
                        proj = p
                if layer_idx is None or proj is None:
                    continue
                key = (layer_idx, proj)
                g2 = param.grad.detach().pow(2).flatten()
                fisher_accum[key] = fisher_accum.get(key, torch.zeros_like(g2)) + g2
        batches_done += 1
    model.zero_grad()
    result = {}
    for key, accum in fisher_accum.items():
        result[key] = float((accum / batches_done).max().item()) if batches_done > 0 else 0.0
    for i in range(12):
        for proj in ("query", "value"):
            result.setdefault((i, proj), 0.0)
    return result

def global_lmax(lmax_dict):
    vals = [v for v in lmax_dict.values() if v > 0]
    return max(vals) if vals else 0.0

print("Fisher/curvature utilities ready.")

# ============================================================================
# Cell 6 -- Result Persistence Helpers (resume-safe)
# ============================================================================
def load_if_exists(run_name, save_dir):
    path = os.path.join(save_dir, f"{run_name}_result.json")
    if os.path.exists(path):
        with open(path) as f:
            result = json.load(f)
        print(f"  {run_name}: already saved (Test Acc {result.get('test_acc', 0):.2f}%) — skipping.")
        return result
    return None

def save_result(run_name, save_dir, result):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{run_name}_result.json"), "w") as f:
        json.dump(result, f, indent=2)

def save_epoch_checkpoint(run_name, save_dir, model, epoch_done, extra_state):
    os.makedirs(save_dir, exist_ok=True)
    meta_path = os.path.join(save_dir, f"{run_name}_progress.json")
    ckpt_path = os.path.join(save_dir, f"{run_name}_progress.pt")
    meta = {"run_name": run_name, "epoch_done": epoch_done, **extra_state}
    torch.save({"model_state": model.state_dict()}, ckpt_path + ".tmp")
    with open(meta_path + ".tmp", "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(meta_path + ".tmp", meta_path)
    os.replace(ckpt_path + ".tmp", ckpt_path)
    print(f"  Checkpoint saved after epoch {epoch_done} -> {meta_path}")

def load_epoch_checkpoint(run_name, save_dir):
    meta_path = os.path.join(save_dir, f"{run_name}_progress.json")
    ckpt_path = os.path.join(save_dir, f"{run_name}_progress.pt")
    if not (os.path.exists(meta_path) and os.path.exists(ckpt_path)):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"  Resuming {run_name} from epoch {meta['epoch_done']}")
    return meta, ckpt

def clear_epoch_checkpoint(run_name, save_dir):
    for suffix in ["_progress.json", "_progress.pt"]:
        p = os.path.join(save_dir, f"{run_name}{suffix}")
        if os.path.exists(p):
            os.remove(p)

print("Persistence helpers ready.")

# ============================================================================
# PHASE 1 -- Fixed-Rank Baselines
# ============================================================================
results_path = os.path.join(PHASE1_DIR, "results.json")
phase1_results = json.load(open(results_path)) if os.path.exists(results_path) else {}

if "Full fine-tune" not in phase1_results:
    model = build_full_model()
    model = train_fixed_model(model, lr=CONFIG["lr_full"], label="Full FT")
    test_acc, test_loss = evaluate(model, test_loader)
    t, _ = count_parameters(model)
    phase1_results["Full fine-tune"] = {"test_acc": test_acc, "test_loss": test_loss, "trainable_params": t}
    json.dump(phase1_results, open(results_path, "w"), indent=2)
    print(f"Full fine-tune — Test Acc {test_acc:.2f}%")
    del model; torch.cuda.empty_cache()
else:
    print(f"Full fine-tune already done: {phase1_results['Full fine-tune']['test_acc']:.2f}%")

if "Frozen backbone" not in phase1_results:
    model = build_frozen_model()
    model = train_fixed_model(model, lr=CONFIG["lr_frozen"], label="Frozen")
    test_acc, test_loss = evaluate(model, test_loader)
    t, _ = count_parameters(model)
    phase1_results["Frozen backbone"] = {"test_acc": test_acc, "test_loss": test_loss, "trainable_params": t}
    json.dump(phase1_results, open(results_path, "w"), indent=2)
    print(f"Frozen backbone — Test Acc {test_acc:.2f}%")
    del model; torch.cuda.empty_cache()
else:
    print(f"Frozen backbone already done: {phase1_results['Frozen backbone']['test_acc']:.2f}%")

for r in CONFIG["lora_ranks"]:
    key = f"LoRA r{r}"
    ckpt_path = os.path.join(PHASE1_DIR, f"lora_r{r}_full.pt")
    if key in phase1_results and os.path.exists(ckpt_path):
        print(f"  r={r:4d} already done: acc={phase1_results[key]['test_acc']:.2f}% — skipping.")
        continue
    print(f"Training LoRA r={r} ({CONFIG['lora_ranks'].index(r)+1}/{len(CONFIG['lora_ranks'])})")
    torch.cuda.empty_cache()
    model = build_lora_model(r)
    t, _ = count_parameters(model)
    model = train_fixed_model(model, lr=CONFIG["lr_lora"], label=f"LoRA r{r}")
    test_acc, test_loss = evaluate(model, test_loader)
    torch.save(model.state_dict(), ckpt_path)
    phase1_results[key] = {"test_acc": test_acc, "test_loss": test_loss,
                            "trainable_params": t, "ckpt_path": ckpt_path}
    json.dump(phase1_results, open(results_path, "w"), indent=2)
    print(f"  r={r} — Test Acc {test_acc:.2f}%  (checkpoint saved)")
    del model; torch.cuda.empty_cache()

print("\nPhase 1 complete.")
print(f"{'Method':22s} {'Trainable Params':>18s} {'Test Acc':>10s}")
print("-" * 55)
for method in ["Full fine-tune", "Frozen backbone"] + [f"LoRA r{r}" for r in CONFIG["lora_ranks"]]:
    if method not in phase1_results:
        continue
    v = phase1_results[method]
    print(f"{method:22s} {v['trainable_params']:>18,} {v['test_acc']:>9.2f}%")

# ============================================================================
# PHASE 2 -- Per-Layer Curvature (Fisher) Profiling
# ============================================================================
N_CURV_BATCHES = 16  # bumped from 8 (used on prior datasets) for stability
# on this much smaller training pool
curv_path = os.path.join(PHASE2_DIR, "curvature_results.json")
curvature_results = json.load(open(curv_path)) if os.path.exists(curv_path) else {}

for r in CONFIG["lora_ranks"]:
    key = f"r{r}"
    ckpt_path = os.path.join(PHASE1_DIR, f"lora_r{r}_full.pt")
    if key in curvature_results:
        print(f"  r={r:4d} curvature already computed — skipping.")
        continue
    if not os.path.exists(ckpt_path):
        print(f"  r={r:4d} checkpoint missing — run Phase 1 first.")
        continue
    torch.cuda.empty_cache()
    model = build_lora_model(r)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    gf = compute_global_fisher(model, probe_loader, n_batches=N_CURV_BATCHES)
    lf = compute_perlayer_fisher(model, probe_loader, n_batches=N_CURV_BATCHES)
    lf_serial = {f"{k[0]}_{k[1]}": v for k, v in lf.items()}
    curvature_results[key] = {**gf, "per_layer": lf_serial}
    json.dump(curvature_results, open(curv_path, "w"), indent=2)
    print(f"  r={r:4d} lambda_max={gf['lambda_max']:.6f}  (saved)")
    del model; torch.cuda.empty_cache()

print("\nPhase 2 curvature profiling complete.")

PHASE2_LAYER_LMAX_R16 = {}
if "r16" in curvature_results:
    for key_str, val in curvature_results["r16"]["per_layer"].items():
        layer_idx, proj = key_str.split("_")
        PHASE2_LAYER_LMAX_R16[(int(layer_idx), proj)] = val
if not PHASE2_LAYER_LMAX_R16:
    raise RuntimeError("r16 per-layer curvature missing — re-run Phase 2 for r=16 before proceeding.")

vals = np.array(list(PHASE2_LAYER_LMAX_R16.values()))
PCT25, PCT50, PCT75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 50)), float(np.percentile(vals, 75))
print(f"Phase 2 per-layer lambda_max (r=16), {len(PHASE2_LAYER_LMAX_R16)} layer-proj pairs")
print(f"  min={vals.min():.6f}  25th={PCT25:.6f}  50th={PCT50:.6f}  75th={PCT75:.6f}  max={vals.max():.6f}")

# ============================================================================
# PHASE 3 -- Global CGRS (fractional check_every/cooldown/grace for fair
# cross-dataset comparison; everything else matches CIFAR-100/SVHN)
# ============================================================================
GLOBAL_RANK_LIST = [3, 5, 6, 10, 12, 16, 30, 52, 64]
G_CHECK_EVERY = CHECK_EVERY
G_COOLDOWN = COOLDOWN
G_PROBE_BATCHES = 16

def run_global_cgrs(run_name, tau, r_init=16, epochs=CONFIG["epochs"], save_dir=PHASE3_DIR):
    print(f"Run: {run_name}  tau={tau:.4f}  r_init={r_init}")
    model = build_lora_model(r_init)
    optimizer = get_optimizer(model, CONFIG["lr_lora"], CONFIG["weight_decay"])
    total_steps = epochs * len(train_loader)
    scheduler = get_scheduler(optimizer, total_steps)
    criterion = nn.CrossEntropyLoss()

    cur_r = r_init
    step = 0
    rank_changes = []
    last_change_step = -G_COOLDOWN

    model.train()
    for epoch in range(epochs):
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            step += 1
            optimizer.zero_grad()
            loss = criterion(model(pixel_values=imgs).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()

            if (step % G_CHECK_EVERY == 0
                    and step >= total_steps * GRACE_FRACTION
                    and step - last_change_step >= G_COOLDOWN):
                gf = compute_global_fisher(model, train_loader, n_batches=G_PROBE_BATCHES)
                lmax = gf["lambda_max"]
                idx = GLOBAL_RANK_LIST.index(cur_r) if cur_r in GLOBAL_RANK_LIST else 0
                if lmax > tau and idx < len(GLOBAL_RANK_LIST) - 1:
                    overshoot = lmax / tau
                    steps_to_jump = min(int(overshoot), len(GLOBAL_RANK_LIST) - 1 - idx)
                    new_r = GLOBAL_RANK_LIST[idx + max(1, steps_to_jump)]
                elif lmax < tau * 0.3 and idx > 0:
                    new_r = GLOBAL_RANK_LIST[idx - 1]
                else:
                    new_r = cur_r
                if new_r != cur_r:
                    cur_lr = float(scheduler.get_last_lr()[0])
                    scaling_ratio = new_r / cur_r
                    compensated_lr = cur_lr * scaling_ratio
                    model = transition_lora_rank(model, cur_r, new_r)
                    optimizer = get_optimizer(model, compensated_lr, CONFIG["weight_decay"])
                    scheduler = get_scheduler(optimizer, max(total_steps - step, 1))
                    rank_changes.append({"step": step, "old_r": cur_r, "new_r": new_r, "lambda": lmax})
                    cur_r = new_r
                    last_change_step = step
                    print(f"  Step {step}: rank {rank_changes[-1]['old_r']} -> {new_r}  "
                          f"(lambda={lmax:.5f})  lr {cur_lr:.2e}->{compensated_lr:.2e}")

        val_acc, _ = evaluate(model, val_loader)
        print(f"  Epoch {epoch+1}/{epochs} done. Val Acc {val_acc:.2f}%  Current rank {cur_r}")

    test_acc, test_loss = evaluate(model, test_loader)
    result = {"run_name": run_name, "tau": tau, "final_rank": cur_r, "test_acc": test_acc,
              "test_loss": test_loss, "total_lora_params": count_trainable(model),
              "n_rank_changes": len(rank_changes), "rank_changes": rank_changes}
    save_result(run_name, save_dir, result)
    print(f"  {run_name} COMPLETE — Test Acc {test_acc:.2f}%  Final Rank {cur_r}")
    del model; torch.cuda.empty_cache()
    return result

print("Phase 3 global CGRS function ready.")

PHASE3_TAUS = {
    "CGRS_Global_tau_aggressive": PCT25,
    "CGRS_Global_tau_moderate": PCT50,
    "CGRS_Global_tau_conservative": PCT75,
}
phase3_results = {}
for run_name, tau in PHASE3_TAUS.items():
    result = load_if_exists(run_name, PHASE3_DIR)
    if result is None:
        result = run_global_cgrs(run_name, tau=tau, r_init=16, epochs=CONFIG["epochs"])
    phase3_results[run_name] = result

print("\nPhase 3 complete.")
print(f"{'Run':32s} {'Tau':>10s} {'Final Rank':>10s} {'Test Acc':>10s}")
print("-" * 66)
for name, res in phase3_results.items():
    print(f"{name:32s} {res['tau']:>10.5f} {res['final_rank']:>10d} {res['test_acc']:>9.2f}%")

# ============================================================================
# PHASE 4 -- Per-Layer CGRS (TRUE per-layer via rank_pattern + in-place
# resize preserving optimizer momentum). Same r_init/thresholds/rank-list
# methodology as CIFAR-100/SVHN; only check_every/cooldown are fractional.
# ============================================================================
P4 = {
    "r_init": 30, "r_min": 3, "r_max": 64,
    "rank_list": [3, 5, 6, 10, 12, 16, 30, 52, 64],
    "check_every": CHECK_EVERY, "cooldown": COOLDOWN, "probe_batches": 16,
    "epochs": 20, "lr": 5e-4, "weight_decay": 0.01, "lora_alpha": 16,
}

def live_warmup_calibration(r_init, warmup_steps, n_probe=16, seed=SEED):
    torch.manual_seed(seed)
    model = build_lora_model(r_init)
    optimizer = get_optimizer(model, P4["lr"], P4["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    model.train()
    step = 0
    for imgs, labels in train_loader:
        if step >= warmup_steps:
            break
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(pixel_values=imgs).logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        step += 1
    live_lmax = compute_perlayer_fisher(model, train_loader, n_probe)
    del model; torch.cuda.empty_cache()
    vals = np.array(list(live_lmax.values()))
    return {"per_layer": live_lmax, "pct25": float(np.percentile(vals, 25)),
            "pct50": float(np.percentile(vals, 50)), "pct75": float(np.percentile(vals, 75))}

# Warm-up steps scaled the same way as check_every/cooldown -- roughly the
# first "grace period" worth of steps, capped so it never exceeds an epoch.
WARMUP_STEPS = min(len(train_loader), max(20, round(GRACE_FRACTION * TOTAL_STEPS_PER_RUN)))

live_cal_path = os.path.join(PHASE4_DIR, "live_calibration.json")
if os.path.exists(live_cal_path):
    saved = json.load(open(live_cal_path))
    LIVE_CAL = {"per_layer": {eval(k): v for k, v in saved["per_layer"].items()},
                "pct25": saved["pct25"], "pct50": saved["pct50"], "pct75": saved["pct75"]}
    print("Loaded existing live calibration from disk.")
else:
    LIVE_CAL = live_warmup_calibration(r_init=P4["r_init"], warmup_steps=WARMUP_STEPS, n_probe=P4["probe_batches"])
    json.dump({"per_layer": {str(k): v for k, v in LIVE_CAL["per_layer"].items()},
               "pct25": LIVE_CAL["pct25"], "pct50": LIVE_CAL["pct50"], "pct75": LIVE_CAL["pct75"]},
              open(live_cal_path, "w"), indent=2)
    print(f"Computed fresh live calibration ({WARMUP_STEPS} warm-up steps) and saved to disk.")

print(f"Live calibration (r_init={P4['r_init']}): 25th={LIVE_CAL['pct25']:.6f} "
      f"50th={LIVE_CAL['pct50']:.6f} 75th={LIVE_CAL['pct75']:.6f}")

TAU_C1 = PCT50
TAU_C3_LIVE = dict({k: v * 0.5 for k, v in LIVE_CAL["per_layer"].items()})

def build_lora_model_perlayer(layer_ranks, default_r=16):
    base = ViTForImageClassification.from_pretrained(
        CONFIG["model_name"], num_labels=CONFIG["num_classes"], ignore_mismatched_sizes=True)
    rank_pattern = {}
    for i, r in layer_ranks.items():
        rank_pattern[f"vit.encoder.layer.{i}.attention.attention.query"] = r
        rank_pattern[f"vit.encoder.layer.{i}.attention.attention.value"] = r
    cfg = LoraConfig(r=default_r, rank_pattern=rank_pattern, lora_alpha=CONFIG["lora_alpha"],
                      lora_dropout=CONFIG.get("lora_dropout", 0.1),
                      target_modules=CONFIG.get("target_modules", ["query", "value"]), bias="none")
    model = get_peft_model(base, cfg).to(device)
    expected = sum(3072 * r for r in layer_ranks.values())
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if actual != expected:
        raise RuntimeError(f"rank_pattern mismatch: expected {expected:,}, got {actual:,}")
    return model

def get_lora_module(model, layer_idx, proj):
    target_suffix = f"layer.{layer_idx}.attention.attention.{proj}"
    for name, module in model.named_modules():
        if name.endswith(target_suffix) and hasattr(module, "lora_A"):
            return module
    raise RuntimeError(f"Could not find LoRA module for layer {layer_idx} proj {proj}")

def resize_layer_inplace(model, optimizer, layer_idx, new_r, lora_alpha=CONFIG["lora_alpha"]):
    param_group = optimizer.param_groups[0]
    for proj in ["query", "value"]:
        module = get_lora_module(model, layer_idx, proj)
        old_A_param = module.lora_A["default"].weight
        old_B_param = module.lora_B["default"].weight
        old_r = old_A_param.shape[0]
        in_features, out_features = old_A_param.shape[1], old_B_param.shape[0]
        if new_r == old_r:
            continue
        A, B = old_A_param.data.clone(), old_B_param.data.clone()
        if new_r > old_r:
            new_A = torch.zeros(new_r, in_features, device=A.device); new_A[:old_r] = A
            new_B = torch.zeros(out_features, new_r, device=B.device); new_B[:, :old_r] = B
        else:
            W = B @ A
            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                sqrt_S = torch.sqrt(S[:new_r].clamp(min=0.0))
                new_B = U[:, :new_r] * sqrt_S
                new_A = Vh[:new_r] * sqrt_S.unsqueeze(1)
            except RuntimeError:
                new_A, new_B = A[:new_r], B[:, :new_r]
        if old_A_param in optimizer.state:
            del optimizer.state[old_A_param]
        if old_B_param in optimizer.state:
            del optimizer.state[old_B_param]
        param_group["params"] = [p for p in param_group["params"]
                                  if p is not old_A_param and p is not old_B_param]
        module.lora_A["default"] = nn.Linear(in_features, new_r, bias=False).to(A.device)
        module.lora_A["default"].weight = nn.Parameter(new_A)
        module.lora_B["default"] = nn.Linear(new_r, out_features, bias=False).to(B.device)
        module.lora_B["default"].weight = nn.Parameter(new_B)
        module.r["default"] = new_r
        module.scaling["default"] = lora_alpha / new_r
        param_group["params"].append(module.lora_A["default"].weight)
        param_group["params"].append(module.lora_B["default"].weight)
    return model

def run_perlayer_threshold_cgrs(run_name, tau_config, r_init=P4["r_init"], epochs=P4["epochs"],
                                 save_dir=PHASE4_DIR):
    is_per_layer_tau = isinstance(tau_config, dict)
    RANKLIST, RMAX = P4["rank_list"], P4["r_max"]
    K, COOLDOWN_P4, NPROBE = P4["check_every"], P4["cooldown"], P4["probe_batches"]
    total_steps = epochs * len(train_loader)

    resumed = load_epoch_checkpoint(run_name, save_dir)
    if resumed:
        meta, ckpt = resumed
        start_epoch = meta["epoch_done"]
        layer_ranks = {int(k): v for k, v in meta["layer_ranks"].items()}
        model = build_lora_model_perlayer(layer_ranks)
        model.load_state_dict(ckpt["model_state"])
        cooldown_left = {int(k): v for k, v in meta["cooldown_left"].items()}
        rank_changes = meta["rank_changes"]
        optimizer = get_optimizer(model, meta["cur_lr"], P4["weight_decay"])
        scheduler = get_scheduler(optimizer, max(total_steps - meta["step"], 1))
        step = meta["step"]
    else:
        start_epoch = 0
        layer_ranks = {i: r_init for i in range(12)}
        model = build_lora_model_perlayer(layer_ranks)
        cooldown_left = {i: 0 for i in range(12)}
        rank_changes = []
        optimizer = get_optimizer(model, P4["lr"], P4["weight_decay"])
        scheduler = get_scheduler(optimizer, total_steps)
        step = 0

    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(start_epoch, epochs):
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            step += 1
            optimizer.zero_grad()
            loss = criterion(model(pixel_values=imgs).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()

            if step % K == 0 and step >= total_steps * GRACE_FRACTION:
                lmax_dict = compute_perlayer_fisher(model, train_loader, NPROBE)
                for i in range(12):
                    if cooldown_left[i] > 0:
                        cooldown_left[i] = max(0, cooldown_left[i] - K)
                        continue
                    cur_layer_r = layer_ranks[i]
                    lay_lmax = max(lmax_dict.get((i, "query"), 0.0), lmax_dict.get((i, "value"), 0.0))
                    if is_per_layer_tau:
                        tau_i = max(tau_config.get((i, "query"), 0.0), tau_config.get((i, "value"), 0.0))
                    else:
                        tau_i = tau_config
                    if lay_lmax > tau_i and cur_layer_r < RMAX:
                        idx = RANKLIST.index(cur_layer_r) if cur_layer_r in RANKLIST else 0
                        overshoot = lay_lmax / tau_i if tau_i > 0 else 1
                        steps_to_jump = min(int(overshoot), len(RANKLIST) - 1 - idx)
                        new_r = RANKLIST[idx + max(1, steps_to_jump)]
                        if new_r != cur_layer_r:
                            model = resize_layer_inplace(model, optimizer, i, new_r)
                            layer_ranks[i] = new_r
                            cooldown_left[i] = COOLDOWN_P4
                            rank_changes.append({"step": step, "layer": i, "old_r": cur_layer_r,
                                                  "new_r": new_r, "lambda": lay_lmax, "tau": tau_i})
                            print(f"  Step {step}: Layer {i}  {cur_layer_r} -> {new_r}  "
                                  f"(in-place, other layers untouched)")

        val_acc, _ = evaluate(model, val_loader)
        avg_r = float(np.mean(list(layer_ranks.values())))
        print(f"  Epoch {epoch+1}/{epochs} done. Val Acc {val_acc:.2f}%  Avg Rank {avg_r:.1f}")
        save_epoch_checkpoint(run_name, save_dir, model, epoch + 1, {
            "layer_ranks": layer_ranks, "cooldown_left": cooldown_left,
            "rank_changes": rank_changes, "step": step,
            "cur_lr": float(scheduler.get_last_lr()[0]),
        })

    test_acc, test_loss = evaluate(model, test_loader)
    avg_rank = float(np.mean(list(layer_ranks.values())))
    actual_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = {"run_name": run_name, "tau": "per-layer" if is_per_layer_tau else float(tau_config),
              "r_init": r_init, "test_acc": test_acc, "test_loss": test_loss, "avg_rank": avg_rank,
              "final_ranks": {str(k): v for k, v in layer_ranks.items()},
              "total_lora_params": actual_params, "n_rank_changes": len(rank_changes),
              "rank_changes": rank_changes}
    save_result(run_name, save_dir, result)
    clear_epoch_checkpoint(run_name, save_dir)
    print(f"  {run_name} COMPLETE — Test Acc {test_acc:.2f}%  Avg Rank {avg_rank:.2f}  "
          f"Actual Params {actual_params:,}")
    del model; torch.cuda.empty_cache()
    return result

print("Per-layer threshold-based CGRS function ready.")

result = load_if_exists("PL_C1", PHASE4_DIR)
if result is None:
    result = run_perlayer_threshold_cgrs("PL_C1", tau_config=TAU_C1, r_init=P4["r_init"], epochs=P4["epochs"])
phase4_results = {"PL_C1": result}

result = load_if_exists("PL_C3_Live", PHASE4_DIR)
if result is None:
    result = run_perlayer_threshold_cgrs("PL_C3_Live", tau_config=TAU_C3_LIVE, r_init=P4["r_init"], epochs=P4["epochs"])
phase4_results["PL_C3_Live"] = result

def run_ordinal_cgrs(run_name, top_k, protected_layers=(4, 5, 6, 7), protected_min_rank=16,
                      r_init=P4["r_init"], epochs=P4["epochs"], save_dir=PHASE4_DIR):
    RANKLIST, RMAX = P4["rank_list"], P4["r_max"]
    K, COOLDOWN_P4, NPROBE = P4["check_every"], P4["cooldown"], P4["probe_batches"]
    total_steps = epochs * len(train_loader)

    resumed = load_epoch_checkpoint(run_name, save_dir)
    if resumed:
        meta, ckpt = resumed
        start_epoch = meta["epoch_done"]
        layer_ranks = {int(k): v for k, v in meta["layer_ranks"].items()}
        model = build_lora_model_perlayer(layer_ranks)
        model.load_state_dict(ckpt["model_state"])
        cooldown_left = {int(k): v for k, v in meta["cooldown_left"].items()}
        rank_changes = meta["rank_changes"]
        optimizer = get_optimizer(model, meta["cur_lr"], P4["weight_decay"])
        scheduler = get_scheduler(optimizer, max(total_steps - meta["step"], 1))
        step = meta["step"]
    else:
        start_epoch = 0
        layer_ranks = {i: r_init for i in range(12)}
        model = build_lora_model_perlayer(layer_ranks)
        cooldown_left = {i: 0 for i in range(12)}
        rank_changes = []
        optimizer = get_optimizer(model, P4["lr"], P4["weight_decay"])
        scheduler = get_scheduler(optimizer, total_steps)
        step = 0

    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(start_epoch, epochs):
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            step += 1
            optimizer.zero_grad()
            loss = criterion(model(pixel_values=imgs).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()

            if step % K == 0 and step >= total_steps * GRACE_FRACTION:
                lmax_dict = compute_perlayer_fisher(model, train_loader, NPROBE)
                layer_scores = {i: max(lmax_dict.get((i, "query"), 0.0), lmax_dict.get((i, "value"), 0.0))
                                 for i in range(12)}
                ranked = sorted(range(12), key=lambda i: layer_scores[i], reverse=True)
                top_layers = set(ranked[:top_k])
                global_median = float(np.median(list(layer_scores.values()))) or 1e-9

                for i in range(12):
                    if cooldown_left[i] > 0:
                        cooldown_left[i] = max(0, cooldown_left[i] - K)
                        continue
                    cur_layer_r = layer_ranks[i]
                    needs_floor = i in protected_layers and cur_layer_r < protected_min_rank
                    should_expand = (i in top_layers or needs_floor) and cur_layer_r < RMAX
                    if not should_expand:
                        continue
                    idx = RANKLIST.index(cur_layer_r) if cur_layer_r in RANKLIST else 0
                    if needs_floor:
                        candidates = [r for r in RANKLIST if r >= protected_min_rank]
                        new_r = min(candidates) if candidates else RANKLIST[-1]
                    else:
                        overshoot = layer_scores[i] / global_median
                        steps_to_jump = min(int(overshoot), len(RANKLIST) - 1 - idx)
                        new_r = RANKLIST[idx + max(1, steps_to_jump)]
                    if new_r != cur_layer_r:
                        model = resize_layer_inplace(model, optimizer, i, new_r)
                        layer_ranks[i] = new_r
                        cooldown_left[i] = COOLDOWN_P4
                        rank_changes.append({"step": step, "layer": i, "old_r": cur_layer_r,
                                              "new_r": new_r, "reason": "floor" if needs_floor else "topk"})
                        print(f"  Step {step}: Layer {i} ({'floor' if needs_floor else 'top-k'})  "
                              f"{cur_layer_r} -> {new_r}  (in-place, other layers untouched)")

        val_acc, _ = evaluate(model, val_loader)
        avg_r = float(np.mean(list(layer_ranks.values())))
        print(f"  Epoch {epoch+1}/{epochs} done. Val Acc {val_acc:.2f}%  Avg Rank {avg_r:.1f}")
        save_epoch_checkpoint(run_name, save_dir, model, epoch + 1, {
            "layer_ranks": layer_ranks, "cooldown_left": cooldown_left,
            "rank_changes": rank_changes, "step": step,
            "cur_lr": float(scheduler.get_last_lr()[0]),
        })

    test_acc, test_loss = evaluate(model, test_loader)
    avg_rank = float(np.mean(list(layer_ranks.values())))
    actual_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = {"run_name": run_name, "top_k": top_k, "protected_layers": list(protected_layers),
              "protected_min_rank": protected_min_rank, "r_init": r_init, "test_acc": test_acc,
              "test_loss": test_loss, "avg_rank": avg_rank,
              "final_ranks": {str(k): v for k, v in layer_ranks.items()},
              "total_lora_params": actual_params, "n_rank_changes": len(rank_changes),
              "rank_changes": rank_changes}
    save_result(run_name, save_dir, result)
    clear_epoch_checkpoint(run_name, save_dir)
    print(f"  {run_name} COMPLETE — Test Acc {test_acc:.2f}%  Avg Rank {avg_rank:.2f}  "
          f"Actual Params {actual_params:,}")
    del model; torch.cuda.empty_cache()
    return result

print("Ordinal CGRS function (with protected-floor fix) ready.")

result = load_if_exists("OrdinalK4_Fixed", PHASE4_DIR)
if result is None:
    result = run_ordinal_cgrs("OrdinalK4_Fixed", top_k=4, protected_layers=(4, 5, 6, 7), protected_min_rank=16)
phase4_results["OrdinalK4_Fixed"] = result

result = load_if_exists("OrdinalK5", PHASE4_DIR)
if result is None:
    result = run_ordinal_cgrs("OrdinalK5", top_k=5, protected_layers=(4, 5, 6, 7), protected_min_rank=16)
phase4_results["OrdinalK5"] = result

# ============================================================================
# Correlation Validation
# ============================================================================
def check_correlation(result, reference_lmax, label):
    final_ranks = result["final_ranks"]
    ref_vals, fr_vals = [], []
    for (layer, proj), lmax_val in reference_lmax.items():
        fr_vals.append(final_ranks.get(str(layer), P4["r_init"]))
        ref_vals.append(lmax_val)
    if len(set(fr_vals)) == 1:
        print(f"{label:15s} DEGENERATE (constant rank vector — correlation undefined)")
        return None
    pear_r, pear_p = pearsonr(ref_vals, fr_vals)
    spear_r, spear_p = spearmanr(ref_vals, fr_vals)
    verdict = ("POSITIVE, significant" if pear_r > 0 and spear_p < 0.05
               else "positive, not significant" if pear_r > 0 else "NEGATIVE (anomaly)")
    print(f"{label:15s} vs {'static' if reference_lmax is PHASE2_LAYER_LMAX_R16 else 'live':6s}  "
          f"Pearson r={pear_r:.4f} (p={pear_p:.4f})  Spearman r={spear_r:.4f} (p={spear_p:.4f})  [{verdict}]")
    return pear_r, pear_p, spear_r, spear_p

print("Correlation vs STATIC Phase 2 lambda_max (r=16):")
for name, res in phase4_results.items():
    check_correlation(res, PHASE2_LAYER_LMAX_R16, name)

print("\nCorrelation vs LIVE-calibrated lambda_max:")
for name, res in phase4_results.items():
    check_correlation(res, LIVE_CAL["per_layer"], name)

# ============================================================================
# Final Cross-Phase Comparison Table
# ============================================================================
print(f"{'Method':30s} {'Type':10s} {'Avg Rank':>9s} {'Test Acc':>9s} {'LoRA Params':>13s}")
print("=" * 75)
for r in [3, 16, 64]:
    key = f"LoRA r{r}"
    if key in phase1_results:
        v = phase1_results[key]
        print(f"{'Fixed r'+str(r):30s} {'Static':10s} {r:>9d} {v['test_acc']:>8.2f}% {v['trainable_params']:>13,}")
if "Full fine-tune" in phase1_results:
    v = phase1_results["Full fine-tune"]
    print(f"{'Full fine-tune':30s} {'Static':10s} {'-':>9s} {v['test_acc']:>8.2f}% {v['trainable_params']:>13,}")
print("-" * 75)
for name, res in phase3_results.items():
    print(f"{name:30s} {'Global':10s} {res['final_rank']:>9d} {res['test_acc']:>8.2f}% {res['total_lora_params']:>13,}")
print("-" * 75)
for name, res in phase4_results.items():
    print(f"{name:30s} {'Per-Layer':10s} {res['avg_rank']:>9.2f} {res['test_acc']:>8.2f}% {res['total_lora_params']:>13,}")
print("=" * 75)

# ============================================================================
# Save Complete Cross-Phase Results JSON
# ============================================================================
all_results = {
    "dataset": "Flowers102",
    "check_every": CHECK_EVERY, "cooldown": COOLDOWN, "total_steps_per_run": TOTAL_STEPS_PER_RUN,
    "phase1": phase1_results,
    "phase2_layer_lmax_r16": {f"{k[0]}_{k[1]}": v for k, v in PHASE2_LAYER_LMAX_R16.items()},
    "phase2_percentiles": {"pct25": PCT25, "pct50": PCT50, "pct75": PCT75},
    "phase3": phase3_results,
    "phase4": phase4_results,
    "phase4_live_calibration": {"per_layer": {f"{k[0]}_{k[1]}": v for k, v in LIVE_CAL["per_layer"].items()},
                                 "pct25": LIVE_CAL["pct25"], "pct50": LIVE_CAL["pct50"], "pct75": LIVE_CAL["pct75"]},
}
final_path = os.path.join(PROJECT_BASE, "all_results_complete.json")
json.dump(all_results, open(final_path, "w"), indent=2)
print(f"Saved complete cross-phase results to {final_path}")
