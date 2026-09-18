"""
Bundles all 5 accuracy fixes for Phase 4:
  1. r_init: 16 -> 30 (more baseline capacity per layer)
  2. TAU_C1: 75th pct -> 50th pct (easier expansion trigger)
     TAU_C3_LIVE: halved (easier expansion trigger)
  3. cooldown: -> 200 (no longer needs to be inflated; LR-compounding
     bug that required 400 is gone now that optimizer is never rebuilt)
  4. Bigger single-step jumps when curvature strongly overshoots tau
  5. In-place single-layer resize (confirmed via test_inplace_resize.py) --
     replaces full-model-rebuild-per-transition, so all OTHER layers'
     Adam momentum is preserved across every rank change.

Run with:
    python apply_patches_v4.py
"""
import re
import shutil

TARGET = "CGRS_CIFAR100_Rebuild_AllPhases.py"
BACKUP = TARGET + ".before_patch_v4_backup"

with open(TARGET, "r", encoding="utf-8") as f:
    src = f.read()

all_ok = True


def replace_function_block(src, start_marker, end_marker, new_code, label):
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


def replace_dict_literal(src, var_name, new_dict_source, label):
    """Finds 'var_name = {' and the matching closing brace via bracket
    counting (robust to internal formatting), replaces the whole literal."""
    pattern = re.compile(rf"{re.escape(var_name)}\s*=\s*\{{")
    match = pattern.search(src)
    if not match:
        print(f"[FAIL] {label}: could not find '{var_name} = {{'")
        return src, False
    start = match.start()
    brace_start = src.index("{", match.start())
    depth = 0
    i = brace_start
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    if depth != 0:
        print(f"[FAIL] {label}: brace matching failed")
        return src, False
    end = i + 1
    new_src = src[:start] + new_dict_source + src[end:]
    print(f"[OK]   {label}: replaced.")
    return new_src, True


def replace_assignment_line(src, var_name, new_rhs, label):
    pattern = re.compile(rf"^{re.escape(var_name)}\s*=.*$", re.MULTILINE)
    matches = list(pattern.finditer(src))
    if len(matches) != 1:
        print(f"[FAIL] {label}: expected exactly 1 match for '{var_name} = ...', found {len(matches)}")
        return src, False
    m = matches[0]
    new_src = src[:m.start()] + f"{var_name} = {new_rhs}" + src[m.end():]
    print(f"[OK]   {label}: replaced.")
    return new_src, True


# ── Fix 1 & 3: P4 config dict (r_init -> 30, cooldown -> 200) ──────────────
NEW_P4_DICT = '''P4 = {
    "r_init": 30,
    "r_min": 3,
    "r_max": 64,
    "rank_list": [3, 5, 6, 10, 12, 16, 30, 52, 64],
    "check_every": 200,
    "cooldown": 200,
    "probe_batches": 16,
    "epochs": 3,
    "lr": 5e-4,
    "weight_decay": 0.01,
    "lora_alpha": 16,
}'''
src, ok = replace_dict_literal(src, "P4", NEW_P4_DICT, "Fix 1+3: P4 config (r_init=30, cooldown=200)")
all_ok = all_ok and ok

# ── Fix 2: loosen thresholds ─────────────────────────────────────────────
src, ok = replace_assignment_line(src, "TAU_C1", "PCT50", "Fix 2a: TAU_C1 -> PCT50 (was PCT75)")
all_ok = all_ok and ok

src, ok = replace_assignment_line(
    src, "TAU_C3_LIVE",
    'dict({k: v * 0.5 for k, v in LIVE_CAL["per_layer"].items()})',
    "Fix 2b: TAU_C3_LIVE halved"
)
all_ok = all_ok and ok

# ── Fix 4 & 5: rewrite the whole Phase 4 threshold block ────────────────────
NEW_THRESHOLD_BLOCK = '''def build_lora_model_perlayer(layer_ranks, default_r=16):
    base = ViTForImageClassification.from_pretrained(
        CONFIG["model_name"], num_labels=CONFIG["num_classes"], ignore_mismatched_sizes=True)
    rank_pattern = {}
    for i, r in layer_ranks.items():
        rank_pattern[f"vit.encoder.layer.{i}.attention.attention.query"] = r
        rank_pattern[f"vit.encoder.layer.{i}.attention.attention.value"] = r
    cfg = LoraConfig(
        r=default_r,
        rank_pattern=rank_pattern,
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG.get("lora_dropout", 0.1),
        target_modules=CONFIG.get("target_modules", ["query", "value"]),
        bias="none",
    )
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
    """Resizes ONE layer's query+value LoRA adapters in-place. Every other
    layer's Parameter objects -- and their Adam momentum in optimizer.state --
    are left completely untouched (verified via test_inplace_resize.py)."""
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
            new_A = torch.zeros(new_r, in_features, device=A.device)
            new_A[:old_r] = A
            new_B = torch.zeros(out_features, new_r, device=B.device)
            new_B[:, :old_r] = B
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
                        overshoot = lay_lmax / tau_i if tau_i > 0 else 1
                        steps_to_jump = min(int(overshoot), len(RANKLIST) - 1 - idx)
                        new_r = RANKLIST[idx + max(1, steps_to_jump)]
                        if new_r != cur_layer_r:
                            model = resize_layer_inplace(model, optimizer, i, new_r)
                            layer_ranks[i] = new_r
                            cooldown_left[i] = COOLDOWN
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
    del model
    torch.cuda.empty_cache()
    return result'''

src, ok = replace_function_block(
    src,
    start_marker="def build_lora_model_perlayer(",
    end_marker='print("Per-layer threshold-based CGRS function ready.")',
    new_code=NEW_THRESHOLD_BLOCK,
    label="Fix 4+5: threshold block (in-place resize, bigger jumps)",
)
all_ok = all_ok and ok

# ── Fix 4 & 5: rewrite run_ordinal_cgrs the same way ────────────────────────
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
                        cooldown_left[i] = COOLDOWN
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
    del model
    torch.cuda.empty_cache()
    return result'''

src, ok = replace_function_block(
    src,
    start_marker="def run_ordinal_cgrs(",
    end_marker='print("Ordinal CGRS function (with protected-floor fix) ready.")',
    new_code=NEW_ORDINAL_BLOCK,
    label="Fix 4+5: run_ordinal_cgrs (in-place resize, bigger jumps)",
)
all_ok = all_ok and ok

# ── Write out only if every replacement succeeded ───────────────────────────
shutil.copy(TARGET, BACKUP)
print(f"\\nBackup saved to {BACKUP}")

if all_ok:
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"\\nAll 5 fixes applied successfully. {TARGET} updated.")
    print("Verify syntax:")
    print(f"    python -m py_compile {TARGET}")
else:
    print("\\nOne or more replacements FAILED. The file was NOT modified.")
    print("Send me the exact [FAIL] lines above so I can fix the matching.")
