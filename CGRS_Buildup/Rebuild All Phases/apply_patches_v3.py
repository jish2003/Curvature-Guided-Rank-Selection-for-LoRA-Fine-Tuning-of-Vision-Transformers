"""
Rebuilds Phase 4 with GENUINE per-layer independent LoRA ranks using PEFT's
rank_pattern (confirmed working via test_rank_pattern.py). Replaces the old
"rebuild everyone at one uniform rank" logic with true heterogeneous ranks.

Run with:
    python apply_patches_v3.py
"""
import shutil

TARGET = "CGRS_CIFAR100_Rebuild_AllPhases.py"
BACKUP = TARGET + ".before_patch_v3_backup"

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

# ── run_perlayer_threshold_cgrs — full rebuild with TRUE per-layer ranks ────
NEW_THRESHOLD_BLOCK = '''def build_lora_model_perlayer(layer_ranks, default_r=16):
    """layer_ranks: dict {layer_idx (0-11): rank}. Same rank applied to both
    query and value adapters of that layer, but each layer is independent
    from every other layer -- this is what was missing before."""
    base = ViTForImageClassification.from_pretrained(
        CONFIG["model_name"], num_labels=CONFIG["num_classes"], ignore_mismatched_sizes=True)
    rank_pattern = {}
    for i, r in layer_ranks.items():
        rank_pattern[f"vit.encoder.layer.{i}.attention.attention.query"] = r
        rank_pattern[f"vit.encoder.layer.{i}.attention.attention.value"] = r
    cfg = LoraConfig(
        r=default_r,
        rank_pattern=rank_pattern,
        lora_alpha=P4["lora_alpha"],
        lora_dropout=CONFIG.get("lora_dropout", 0.1),
        target_modules=CONFIG.get("target_modules", ["query", "value"]),
        bias="none",
    )
    model = get_peft_model(base, cfg).to(device)
    expected = sum(3072 * r for r in layer_ranks.values())
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if actual != expected:
        raise RuntimeError(
            f"rank_pattern mismatch: expected {expected:,} trainable params, got {actual:,}. "
            f"Do not proceed -- re-run test_rank_pattern.py to diagnose."
        )
    return model


def transition_lora_rank_perlayer(old_model, old_layer_ranks, new_layer_ranks):
    """Transfers every (layer, proj) adapter's OWN current weights into its
    OWN (possibly unchanged) new size. Only the layer(s) whose rank actually
    differs between old_layer_ranks and new_layer_ranks grow or shrink --
    every other layer is a pure same-size copy. This is the fix for the bug
    where every transition silently forced the whole model to one uniform rank."""
    old_params = {n: p.data.clone().cpu() for n, p in old_model.named_parameters()}
    new_model = build_lora_model_perlayer(new_layer_ranks).cpu()
    new_param_dict = dict(new_model.named_parameters())

    for name, old_data in old_params.items():
        if "lora_A" not in name and "lora_B" not in name and name in new_param_dict:
            new_param_dict[name].data.copy_(old_data)

    lora_bases = {}
    for name, data in old_params.items():
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
        base_prefix = name.split("lora_A")[0] if "lora_A" in name else name.split("lora_B")[0]
        entry = lora_bases.setdefault((layer_idx, proj), {"base_prefix": base_prefix})
        if "lora_A" in name:
            entry["A"] = data
        else:
            entry["B"] = data

    for (layer_idx, proj), pair in lora_bases.items():
        if "A" not in pair or "B" not in pair:
            continue
        A, B = pair["A"].float(), pair["B"].float()
        new_r = new_layer_ranks[layer_idx]
        actual_old_r = A.shape[0]
        if new_r == actual_old_r:
            new_A, new_B = A, B
        elif new_r > actual_old_r:
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
        base_prefix = pair["base_prefix"]
        for name, param in new_model.named_parameters():
            if name.startswith(base_prefix) and "lora_A" in name:
                param.data.copy_(new_A.to(param.dtype))
            elif name.startswith(base_prefix) and "lora_B" in name:
                param.data.copy_(new_B.to(param.dtype))

    return new_model.to(device)


def run_perlayer_threshold_cgrs(run_name, tau_config, r_init=P4["r_init"], epochs=P4["epochs"],
                                 save_dir=PHASE4_DIR):
    is_per_layer_tau = isinstance(tau_config, dict)
    RANKLIST, RMAX = P4["rank_list"], P4["r_max"]
    K, COOLDOWN, NPROBE = P4["check_every"], P4["cooldown"], P4["probe_batches"]
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
                            new_layer_ranks = dict(layer_ranks)
                            new_layer_ranks[i] = new_r
                            model = transition_lora_rank_perlayer(model, layer_ranks, new_layer_ranks)
                            layer_ranks = new_layer_ranks
                            cooldown_left[i] = COOLDOWN
                            rank_changes.append({"step": step, "layer": i, "old_r": cur_layer_r,
                                                  "new_r": new_r, "lambda": lay_lmax, "tau": tau_i})
                            optimizer = get_optimizer(model, cur_lr, P4["weight_decay"])
                            scheduler = get_scheduler(optimizer, max(total_steps - step, 1))
                            print(f"  Step {step}: Layer {i}  {cur_layer_r} -> {new_r}  "
                                  f"(only this layer changed, others untouched)")

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
          f"Actual Params {actual_params:,} (true per-layer, not uniform)")
    del model
    torch.cuda.empty_cache()
    return result'''

src, ok = replace_function(
    src,
    start_marker="def run_perlayer_threshold_cgrs(",
    end_marker='print("Per-layer threshold-based CGRS function ready.")',
    new_code=NEW_THRESHOLD_BLOCK,
    label="run_perlayer_threshold_cgrs + new per-layer builder/transition (Phase 4)",
)
all_ok = all_ok and ok

# ── run_ordinal_cgrs — full rebuild with TRUE per-layer ranks ───────────────
NEW_ORDINAL_BLOCK = '''def run_ordinal_cgrs(run_name, top_k, protected_layers=(4, 5, 6, 7), protected_min_rank=16,
                      r_init=P4["r_init"], epochs=P4["epochs"], save_dir=PHASE4_DIR):
    RANKLIST, RMAX = P4["rank_list"], P4["r_max"]
    K, COOLDOWN, NPROBE = P4["check_every"], P4["cooldown"], P4["probe_batches"]
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
                        new_layer_ranks = dict(layer_ranks)
                        new_layer_ranks[i] = new_r
                        model = transition_lora_rank_perlayer(model, layer_ranks, new_layer_ranks)
                        layer_ranks = new_layer_ranks
                        cooldown_left[i] = COOLDOWN
                        rank_changes.append({"step": step, "layer": i, "old_r": cur_layer_r,
                                              "new_r": new_r, "reason": "floor" if needs_floor else "topk"})
                        optimizer = get_optimizer(model, cur_lr, P4["weight_decay"])
                        scheduler = get_scheduler(optimizer, max(total_steps - step, 1))
                        print(f"  Step {step}: Layer {i} ({'floor' if needs_floor else 'top-k'})  "
                              f"{cur_layer_r} -> {new_r}  (only this layer changed, others untouched)")

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
          f"Actual Params {actual_params:,} (true per-layer, not uniform)")
    del model
    torch.cuda.empty_cache()
    return result'''

src, ok = replace_function(
    src,
    start_marker="def run_ordinal_cgrs(",
    end_marker='print("Ordinal CGRS function (with protected-floor fix) ready.")',
    new_code=NEW_ORDINAL_BLOCK,
    label="run_ordinal_cgrs (Phase 4, true per-layer)",
)
all_ok = all_ok and ok

# ── Write out only if every replacement succeeded ───────────────────────────
shutil.copy(TARGET, BACKUP)
print(f"\\nBackup saved to {BACKUP}")

if all_ok:
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"\\nBoth Phase 4 functions rebuilt with TRUE per-layer ranks. {TARGET} updated.")
    print("Verify syntax:")
    print(f"    python -m py_compile {TARGET}")
else:
    print("\\nOne or more replacements FAILED. The file was NOT modified.")
    print("Send me the exact 'def ...' line for the failed function so I can fix the marker.")
