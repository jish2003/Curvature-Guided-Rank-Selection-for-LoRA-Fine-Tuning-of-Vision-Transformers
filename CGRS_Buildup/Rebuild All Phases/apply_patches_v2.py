"""
Robust patcher v2 — replaces entire functions wholesale instead of trying
to match exact text fragments. Finds each function by its "def" line and
the known print(...) line that comes right after it ends, then swaps
everything in between. Much less fragile than fragment-matching.

Run with:
    python apply_patches_v2.py
"""
import shutil

TARGET = "CGRS_CIFAR100_Rebuild_AllPhases.py"
BACKUP = TARGET + ".before_patch_v2_backup"

with open(TARGET, "r", encoding="utf-8") as f:
    src = f.read()

def replace_function(src, start_marker, end_marker, new_code, label):
    start_idx = src.find(start_marker)
    if start_idx == -1:
        print(f"[FAIL] {label}: could not find start marker {start_marker!r}")
        return src, False
    end_idx = src.find(end_marker, start_idx)
    if end_idx == -1:
        print(f"[FAIL] {label}: could not find end marker {end_marker!r} after start")
        return src, False
    new_src = src[:start_idx] + new_code + "\n\n" + src[end_idx:]
    print(f"[OK]   {label}: replaced.")
    return new_src, True

all_ok = True

# ── Phase 3: run_global_cgrs — full replacement ─────────────────────────────
NEW_RUN_GLOBAL_CGRS = '''def run_global_cgrs(run_name, tau, r_init=16, epochs=CONFIG["epochs"], save_dir=PHASE3_DIR):
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
                    and step >= total_steps * 0.15
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
    del model
    torch.cuda.empty_cache()
    return result'''

src, ok = replace_function(
    src,
    start_marker="def run_global_cgrs(",
    end_marker='print("Phase 3 global CGRS function ready.")',
    new_code=NEW_RUN_GLOBAL_CGRS,
    label="run_global_cgrs (Phase 3)",
)
all_ok = all_ok and ok

# ── Phase 4: run_perlayer_threshold_cgrs — full replacement ─────────────────
NEW_RUN_PERLAYER_THRESHOLD_CGRS = '''def run_perlayer_threshold_cgrs(run_name, tau_config, r_init=P4["r_init"], epochs=P4["epochs"],
                                 save_dir=PHASE4_DIR):
    is_per_layer_tau = isinstance(tau_config, dict)
    RANKLIST, RMAX = P4["rank_list"], P4["r_max"]
    K, COOLDOWN, NPROBE = P4["check_every"], P4["cooldown"], P4["probe_batches"]
    total_steps = epochs * len(train_loader)

    resumed = load_epoch_checkpoint(run_name, save_dir)
    if resumed:
        meta, ckpt = resumed
        start_epoch = meta["epoch_done"]
        cur_r = meta["cur_r"]
        model = build_lora_model(cur_r)
        model.load_state_dict(ckpt["model_state"])
        layer_ranks = {int(k): v for k, v in meta["layer_ranks"].items()}
        cooldown_left = {int(k): v for k, v in meta["cooldown_left"].items()}
        rank_changes = meta["rank_changes"]
        optimizer = get_optimizer(model, meta["cur_lr"], P4["weight_decay"])
        scheduler = get_scheduler(optimizer, max(total_steps - meta["step"], 1))
        step = meta["step"]
    else:
        start_epoch, cur_r = 0, r_init
        model = build_lora_model(r_init)
        layer_ranks = {i: r_init for i in range(12)}
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

            if step % K == 0:
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
                        new_r = RANKLIST[min(idx + 1, len(RANKLIST) - 1)]
                        if new_r != cur_layer_r:
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
                            print(f"  Step {step}: Layer {i}  {cur_layer_r} -> {new_r}  "
                                  f"lr {cur_lr:.2e}->{compensated_lr:.2e}")

        val_acc, _ = evaluate(model, val_loader)
        avg_r = float(np.mean(list(layer_ranks.values())))
        print(f"  Epoch {epoch+1}/{epochs} done. Val Acc {val_acc:.2f}%  Avg Rank {avg_r:.1f}")
        save_epoch_checkpoint(run_name, save_dir, model, epoch + 1, {
            "cur_r": cur_r, "layer_ranks": layer_ranks, "cooldown_left": cooldown_left,
            "rank_changes": rank_changes, "step": step,
            "cur_lr": float(scheduler.get_last_lr()[0]),
        })

    test_acc, test_loss = evaluate(model, test_loader)
    avg_rank = float(np.mean(list(layer_ranks.values())))
    result = {"run_name": run_name, "tau": "per-layer" if is_per_layer_tau else float(tau_config),
              "r_init": r_init, "test_acc": test_acc, "test_loss": test_loss, "avg_rank": avg_rank,
              "final_ranks": {str(k): v for k, v in layer_ranks.items()},
              "total_lora_params": count_trainable(model), "n_rank_changes": len(rank_changes),
              "rank_changes": rank_changes}
    save_result(run_name, save_dir, result)
    clear_epoch_checkpoint(run_name, save_dir)
    print(f"  {run_name} COMPLETE — Test Acc {test_acc:.2f}%  Avg Rank {avg_rank:.2f}")
    del model
    torch.cuda.empty_cache()
    return result'''

src, ok = replace_function(
    src,
    start_marker="def run_perlayer_threshold_cgrs(",
    end_marker='print("Per-layer threshold-based CGRS function ready.")',
    new_code=NEW_RUN_PERLAYER_THRESHOLD_CGRS,
    label="run_perlayer_threshold_cgrs (Phase 4)",
)
all_ok = all_ok and ok

# ── Phase 4: run_ordinal_cgrs — full replacement ────────────────────────────
NEW_RUN_ORDINAL_CGRS = '''def run_ordinal_cgrs(run_name, top_k, protected_layers=(4, 5, 6, 7), protected_min_rank=16,
                      r_init=P4["r_init"], epochs=P4["epochs"], save_dir=PHASE4_DIR):
    RANKLIST, RMAX = P4["rank_list"], P4["r_max"]
    K, COOLDOWN, NPROBE = P4["check_every"], P4["cooldown"], P4["probe_batches"]
    total_steps = epochs * len(train_loader)

    resumed = load_epoch_checkpoint(run_name, save_dir)
    if resumed:
        meta, ckpt = resumed
        start_epoch = meta["epoch_done"]
        cur_r = meta["cur_r"]
        model = build_lora_model(cur_r)
        model.load_state_dict(ckpt["model_state"])
        layer_ranks = {int(k): v for k, v in meta["layer_ranks"].items()}
        cooldown_left = {int(k): v for k, v in meta["cooldown_left"].items()}
        rank_changes = meta["rank_changes"]
        optimizer = get_optimizer(model, meta["cur_lr"], P4["weight_decay"])
        scheduler = get_scheduler(optimizer, max(total_steps - meta["step"], 1))
        step = meta["step"]
    else:
        start_epoch, cur_r = 0, r_init
        model = build_lora_model(r_init)
        layer_ranks = {i: r_init for i in range(12)}
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

            if step % K == 0:
                lmax_dict = compute_perlayer_fisher(model, train_loader, NPROBE)
                layer_scores = {i: max(lmax_dict.get((i, "query"), 0.0), lmax_dict.get((i, "value"), 0.0))
                                 for i in range(12)}
                ranked = sorted(range(12), key=lambda i: layer_scores[i], reverse=True)
                top_layers = set(ranked[:top_k])

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
                        new_r = RANKLIST[min(idx + 1, len(RANKLIST) - 1)]
                    if new_r != cur_layer_r:
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
                              f"{cur_layer_r} -> {new_r}  lr {cur_lr:.2e}->{compensated_lr:.2e}")

        val_acc, _ = evaluate(model, val_loader)
        avg_r = float(np.mean(list(layer_ranks.values())))
        print(f"  Epoch {epoch+1}/{epochs} done. Val Acc {val_acc:.2f}%  Avg Rank {avg_r:.1f}")
        save_epoch_checkpoint(run_name, save_dir, model, epoch + 1, {
            "cur_r": cur_r, "layer_ranks": layer_ranks, "cooldown_left": cooldown_left,
            "rank_changes": rank_changes, "step": step,
            "cur_lr": float(scheduler.get_last_lr()[0]),
        })

    test_acc, test_loss = evaluate(model, test_loader)
    avg_rank = float(np.mean(list(layer_ranks.values())))
    result = {"run_name": run_name, "top_k": top_k, "protected_layers": list(protected_layers),
              "protected_min_rank": protected_min_rank, "r_init": r_init, "test_acc": test_acc,
              "test_loss": test_loss, "avg_rank": avg_rank,
              "final_ranks": {str(k): v for k, v in layer_ranks.items()},
              "total_lora_params": count_trainable(model), "n_rank_changes": len(rank_changes),
              "rank_changes": rank_changes}
    save_result(run_name, save_dir, result)
    clear_epoch_checkpoint(run_name, save_dir)
    print(f"  {run_name} COMPLETE — Test Acc {test_acc:.2f}%  Avg Rank {avg_rank:.2f}")
    del model
    torch.cuda.empty_cache()
    return result'''

src, ok = replace_function(
    src,
    start_marker="def run_ordinal_cgrs(",
    end_marker='print("Ordinal CGRS function (with protected-floor fix) ready.")',
    new_code=NEW_RUN_ORDINAL_CGRS,
    label="run_ordinal_cgrs (Phase 4)",
)
all_ok = all_ok and ok

# ── Write out only if every replacement succeeded ───────────────────────────
shutil.copy(TARGET, BACKUP)
print(f"\\nBackup saved to {BACKUP}")

if all_ok:
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"\\nAll 3 functions replaced successfully. {TARGET} has been updated.")
    print("Now verify syntax:")
    print(f"    python -m py_compile {TARGET}")
else:
    print("\\nOne or more replacements FAILED. The file was NOT modified.")
    print("Send me the exact 'def ...' line and the line right after the function's")
    print("closing 'return result' so I can fix the marker text.")
