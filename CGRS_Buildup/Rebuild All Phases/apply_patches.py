"""
Safe patcher for CGRS_CIFAR100_Rebuild_AllPhases.py
Applies 3 code fixes via exact string matching (no manual retyping, no
indentation risk). Aborts without touching the file if any patch doesn't
match exactly once, so you never end up with a half-patched file.

Run with:
    python apply_patches.py
"""
import shutil

TARGET = "CGRS_CIFAR100_Rebuild_AllPhases.py"
BACKUP = TARGET + ".before_patch_backup"

with open(TARGET, "r") as f:
    src = f.read()

patches = []

# ── PATCH A1: alpha/rank scaling compensation — Phase 3 Global CGRS ────────
old_a1 = '''                if new_r != cur_r:
                    cur_lr = float(scheduler.get_last_lr()[0])
                    model = transition_lora_rank(model, cur_r, new_r)
                    optimizer = get_optimizer(model, cur_lr, CONFIG["weight_decay"])
                    scheduler = get_scheduler(optimizer, max(total_steps - step, 1))
                    rank_changes.append({"step": step, "old_r": cur_r, "new_r": new_r, "lambda": lmax})
                    cur_r = new_r
                    last_change_step = step
                    print(f"  Step {step}: rank {rank_changes[-1]['old_r']} -> {new_r}  (lambda={lmax:.5f})")'''

new_a1 = '''                if new_r != cur_r:
                    cur_lr = float(scheduler.get_last_lr()[0])
                    scaling_ratio = new_r / cur_r  # compensates for PEFT's alpha/r shrinkage on rank growth
                    compensated_lr = cur_lr * scaling_ratio
                    model = transition_lora_rank(model, cur_r, new_r)
                    optimizer = get_optimizer(model, compensated_lr, CONFIG["weight_decay"])
                    scheduler = get_scheduler(optimizer, max(total_steps - step, 1))
                    rank_changes.append({"step": step, "old_r": cur_r, "new_r": new_r, "lambda": lmax})
                    cur_r = new_r
                    last_change_step = step
                    print(f"  Step {step}: rank {rank_changes[-1]['old_r']} -> {new_r}  (lambda={lmax:.5f})  lr {cur_lr:.2e}->{compensated_lr:.2e}")'''

patches.append(("A1: alpha/rank scaling comp (Phase 3)", old_a1, new_a1))

# ── PATCH A2: same fix — Phase 4 threshold-based CGRS (PL_C1, PL_C3_Live) ──
old_a2 = '''                        if new_r != cur_layer_r:
                            cur_lr = float(scheduler.get_last_lr()[0])
                            model = transition_lora_rank(model, cur_layer_r, new_r)
                            layer_ranks[i] = new_r
                            cooldown_left[i] = COOLDOWN
                            rank_changes.append({"step": step, "layer": i, "old_r": cur_layer_r,
                                                  "new_r": new_r, "lambda": lay_lmax, "tau": tau_i})
                            optimizer = get_optimizer(model, cur_lr, P4["weight_decay"])
                            scheduler = get_scheduler(optimizer, max(total_steps - step, 1))
                            print(f"  Step {step}: Layer {i}  {cur_layer_r} -> {new_r}")'''

new_a2 = '''                        if new_r != cur_layer_r:
                            cur_lr = float(scheduler.get_last_lr()[0])
                            scaling_ratio = new_r / cur_layer_r
                            compensated_lr = cur_lr * scaling_ratio
                            model = transition_lora_rank(model, cur_layer_r, new_r)
                            layer_ranks[i] = new_r
                            cooldown_left[i] = COOLDOWN
                            rank_changes.append({"step": step, "layer": i, "old_r": cur_layer_r,
                                                  "new_r": new_r, "lambda": lay_lmax, "tau": tau_i})
                            optimizer = get_optimizer(model, compensated_lr, P4["weight_decay"])
                            scheduler = get_scheduler(optimizer, max(total_steps - step, 1))
                            print(f"  Step {step}: Layer {i}  {cur_layer_r} -> {new_r}  lr {cur_lr:.2e}->{compensated_lr:.2e}")'''

patches.append(("A2: alpha/rank scaling comp (Phase 4 threshold)", old_a2, new_a2))

# ── PATCH A3: same fix — Phase 4 ordinal CGRS (OrdinalK4_Fixed, OrdinalK5) ──
old_a3 = '''                    if new_r != cur_layer_r:
                        cur_lr = float(scheduler.get_last_lr()[0])
                        model = transition_lora_rank(model, cur_layer_r, new_r)
                        layer_ranks[i] = new_r
                        cooldown_left[i] = COOLDOWN
                        rank_changes.append({"step": step, "layer": i, "old_r": cur_layer_r,
                                              "new_r": new_r, "reason": "floor" if needs_floor else "topk"})
                        optimizer = get_optimizer(model, cur_lr, P4["weight_decay"])
                        scheduler = get_scheduler(optimizer, max(total_steps - step, 1))
                        print(f"  Step {step}: Layer {i} ({'floor' if needs_floor else 'top-k'})  "
                              f"{cur_layer_r} -> {new_r}")'''

new_a3 = '''                    if new_r != cur_layer_r:
                        cur_lr = float(scheduler.get_last_lr()[0])
                        scaling_ratio = new_r / cur_layer_r
                        compensated_lr = cur_lr * scaling_ratio
                        model = transition_lora_rank(model, cur_layer_r, new_r)
                        layer_ranks[i] = new_r
                        cooldown_left[i] = COOLDOWN
                        rank_changes.append({"step": step, "layer": i, "old_r": cur_layer_r,
                                              "new_r": new_r, "reason": "floor" if needs_floor else "topk"})
                        optimizer = get_optimizer(model, compensated_lr, P4["weight_decay"])
                        scheduler = get_scheduler(optimizer, max(total_steps - step, 1))
                        print(f"  Step {step}: Layer {i} ({'floor' if needs_floor else 'top-k'})  "
                              f"{cur_layer_r} -> {new_r}  lr {cur_lr:.2e}->{compensated_lr:.2e}")'''

patches.append(("A3: alpha/rank scaling comp (Phase 4 ordinal)", old_a3, new_a3))

# ── PATCH B: bigger jumps instead of always stepping one rung — Phase 3 ────
old_b = '''                idx = GLOBAL_RANK_LIST.index(cur_r) if cur_r in GLOBAL_RANK_LIST else 0
                if lmax > tau and idx < len(GLOBAL_RANK_LIST) - 1:
                    new_r = GLOBAL_RANK_LIST[idx + 1]
                elif lmax < tau * 0.3 and idx > 0:
                    new_r = GLOBAL_RANK_LIST[idx - 1]
                else:
                    new_r = cur_r'''

new_b = '''                idx = GLOBAL_RANK_LIST.index(cur_r) if cur_r in GLOBAL_RANK_LIST else 0
                if lmax > tau and idx < len(GLOBAL_RANK_LIST) - 1:
                    overshoot = lmax / tau
                    steps_to_jump = min(int(overshoot), len(GLOBAL_RANK_LIST) - 1 - idx)
                    new_r = GLOBAL_RANK_LIST[idx + max(1, steps_to_jump)]
                elif lmax < tau * 0.3 and idx > 0:
                    new_r = GLOBAL_RANK_LIST[idx - 1]
                else:
                    new_r = cur_r'''

patches.append(("B: bigger single jumps (Phase 3)", old_b, new_b))

# ── PATCH C: grace period scales with total training length — Phase 3 ─────
old_c = '''            if step % G_CHECK_EVERY == 0 and step >= G_CHECK_EVERY * 3 and step - last_change_step >= G_COOLDOWN:'''
new_c = '''            if step % G_CHECK_EVERY == 0 and step >= total_steps * 0.15 and step - last_change_step >= G_COOLDOWN:'''

patches.append(("C: grace period as fraction of total_steps (Phase 3)", old_c, new_c))

# ── Apply all patches, verify each matches exactly once, abort if not ──────
shutil.copy(TARGET, BACKUP)
print(f"Backup saved to {BACKUP}\n")

failed = False
for name, old, new in patches:
    count = src.count(old)
    if count == 1:
        src = src.replace(old, new)
        print(f"[OK]   {name} — applied.")
    elif count == 0:
        print(f"[SKIP] {name} — pattern not found (already patched, or code differs from expected).")
        failed = True
    else:
        print(f"[FAIL] {name} — pattern found {count} times, expected exactly 1. Not applying to avoid ambiguity.")
        failed = True

if failed:
    print("\nOne or more patches did NOT apply cleanly. The file was NOT modified.")
    print(f"Original file is untouched. Backup also exists at {BACKUP} for reference.")
else:
    with open(TARGET, "w") as f:
        f.write(src)
    print(f"\nAll patches applied successfully. {TARGET} has been updated.")
    print("Verify syntax before running:")
    print(f"    python -m py_compile {TARGET}")
