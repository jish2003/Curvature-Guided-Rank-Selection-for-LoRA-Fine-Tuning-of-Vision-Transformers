"""
AdaLoRA Baseline Comparison -- runs against CIFAR-100, SVHN, or Flowers-102
using the SAME model, splits, epochs, and evaluation logic as your existing
CGRS pipelines, so results drop directly into your existing tables.

Usage:
    python CGRS_AdaLoRA_Comparison.py --dataset cifar100
    python CGRS_AdaLoRA_Comparison.py --dataset svhn
    python CGRS_AdaLoRA_Comparison.py --dataset flowers102

Uses peft's built-in AdaLoraConfig / AdaLoraModel -- no custom rank-allocation
code needed. Runs 3 target_r settings per dataset (16, 32, 64) so you get
comparison points against your existing Fixed r16/r64 rows and against your
CGRS methods' typical final average ranks.
"""
import os, json, argparse, random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from peft import AdaLoraConfig, get_peft_model

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=["cifar100", "svhn", "flowers102"])
args = parser.parse_args()
DATASET = args.dataset

# ============================================================================
# Dataset-specific config -- mirrors your existing per-dataset pipelines
# ============================================================================
DATASET_CONFIG = {
    "cifar100": {"num_classes": 100, "epochs": 3, "batch_size": 16,
                 "project_base": "~/CGRS_Project/CIFAR100_Rebuild"},
    "svhn":     {"num_classes": 10,  "epochs": 3, "batch_size": 16,
                 "project_base": "~/CGRS_Project/SVHN_Rebuild"},
    "flowers102": {"num_classes": 102, "epochs": 20, "batch_size": 16,
                    "project_base": "~/CGRS_Project/Flowers102_Rebuild"},
}
DCFG = DATASET_CONFIG[DATASET]

MODEL_NAME = "google/vit-base-patch16-224"
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
TARGET_MODULES = ["query", "value"]
LR = 5e-4
WEIGHT_DECAY = 0.01
TARGET_R_LIST = [16, 32, 64]   # comparison points -- matches your Fixed r16/r64 rows

PROJECT_BASE = os.path.expanduser(DCFG["project_base"])
ADALORA_DIR = os.path.join(PROJECT_BASE, "adalora_results")
DATA_DIR = os.path.join(PROJECT_BASE, "data")
os.makedirs(ADALORA_DIR, exist_ok=True)

print(f"Dataset: {DATASET}  |  Project base: {PROJECT_BASE}")

# ============================================================================
# Data loading -- identical logic to your existing per-dataset scripts
# ============================================================================
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

if DATASET == "cifar100":
    full_train = datasets.CIFAR100(root=DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=transform)
    train_size = 45000
    train_dataset, val_dataset = random_split(
        full_train, [train_size, len(full_train) - train_size],
        generator=torch.Generator().manual_seed(SEED))

elif DATASET == "svhn":
    full_train = datasets.SVHN(root=DATA_DIR, split="train", download=True, transform=transform)
    test_dataset = datasets.SVHN(root=DATA_DIR, split="test", download=True, transform=transform)
    train_size = 68000
    train_dataset, val_dataset = random_split(
        full_train, [train_size, len(full_train) - train_size],
        generator=torch.Generator().manual_seed(SEED))

else:  # flowers102
    official_train = datasets.Flowers102(root=DATA_DIR, split="train", download=True, transform=transform)
    official_val = datasets.Flowers102(root=DATA_DIR, split="val", download=True, transform=transform)
    test_dataset = datasets.Flowers102(root=DATA_DIR, split="test", download=True, transform=transform)
    combined_pool = ConcatDataset([official_train, official_val])
    val_size = 200
    train_dataset, val_dataset = random_split(
        combined_pool, [len(combined_pool) - val_size, val_size],
        generator=torch.Generator().manual_seed(SEED))

NUM_WORKERS = 4
train_loader = DataLoader(train_dataset, batch_size=DCFG["batch_size"], shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=DCFG["batch_size"], shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=DCFG["batch_size"], shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

TOTAL_STEPS = DCFG["epochs"] * len(train_loader)
print(f"Train: {len(train_dataset):,}  Val: {len(val_dataset):,}  Test: {len(test_dataset):,}")
print(f"Steps/epoch: {len(train_loader)}  Total steps/run: {TOTAL_STEPS}")

# ============================================================================
# Model / eval utilities
# ============================================================================
def build_adalora_model(init_r, target_r, total_steps):
    base = ViTForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=DCFG["num_classes"], ignore_mismatched_sizes=True)
    tinit = max(1, int(0.15 * total_steps))     # same grace-period convention as your CGRS runs
    tfinal = max(1, int(0.15 * total_steps))
    delta_t = max(1, int(0.024 * total_steps))  # same check-cadence fraction as your CGRS scripts
    cfg = AdaLoraConfig(
        init_r=init_r, target_r=target_r,
        tinit=tinit, tfinal=tfinal, deltaT=delta_t,
        total_step=total_steps,
        lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES, bias="none",
    )
    return get_peft_model(base, cfg).to(device)

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

def load_if_exists(run_name):
    path = os.path.join(ADALORA_DIR, f"{run_name}_result.json")
    if os.path.exists(path):
        with open(path) as f:
            result = json.load(f)
        print(f"  {run_name}: already saved (Test Acc {result['test_acc']:.2f}%) — skipping.")
        return result
    return None

def save_result(run_name, result):
    with open(os.path.join(ADALORA_DIR, f"{run_name}_result.json"), "w") as f:
        json.dump(result, f, indent=2)

# ============================================================================
# AdaLoRA training loop -- the ONLY difference from a normal LoRA loop is the
# extra update_and_allocate() call after every optimizer step.
# ============================================================================
def train_adalora(target_r, init_r_multiplier=1.5):
    init_r = int(target_r * init_r_multiplier)   # AdaLoRA starts above target and prunes down
    run_name = f"AdaLoRA_target_r{target_r}"

    cached = load_if_exists(run_name)
    if cached:
        return cached

    print(f"Run: {run_name}  init_r={init_r}  target_r={target_r}")
    model = build_adalora_model(init_r, target_r, TOTAL_STEPS)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    model.train()
    for epoch in range(DCFG["epochs"]):
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(pixel_values=imgs).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            # AdaLoRA-specific: performs importance-based rank pruning/reallocation
            model.base_model.update_and_allocate(global_step)
            scheduler.step()
            global_step += 1

        val_acc, _ = evaluate(model, val_loader)
        print(f"  Epoch {epoch+1}/{DCFG['epochs']} done. Val Acc {val_acc:.2f}%")

    test_acc, test_loss = evaluate(model, test_loader)
    actual_params = count_trainable(model)
    result = {
        "run_name": run_name, "dataset": DATASET,
        "init_r": init_r, "target_r": target_r,
        "test_acc": test_acc, "test_loss": test_loss,
        "total_lora_params": actual_params,
    }
    save_result(run_name, result)
    print(f"  {run_name} COMPLETE — Test Acc {test_acc:.2f}%  Params {actual_params:,}")
    del model; torch.cuda.empty_cache()
    return result

# ============================================================================
# Run all target_r settings and print comparison table
# ============================================================================
results = {}
for target_r in TARGET_R_LIST:
    results[f"AdaLoRA_target_r{target_r}"] = train_adalora(target_r)

print(f"\n{'='*70}\nAdaLoRA Comparison — {DATASET}\n{'='*70}")
print(f"{'Run':25s} {'Target r':>10s} {'Test Acc':>10s} {'Params':>12s}")
print("-" * 70)
for name, res in results.items():
    print(f"{name:25s} {res['target_r']:>10d} {res['test_acc']:>9.2f}% {res['total_lora_params']:>12,}")

final_path = os.path.join(ADALORA_DIR, "adalora_all_results.json")
json.dump(results, open(final_path, "w"), indent=2)
print(f"\nSaved to {final_path}")
