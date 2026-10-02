"""
Verifies that a single layer's LoRA rank can be resized IN-PLACE (without
rebuilding the whole model) and that doing so leaves every OTHER layer's
Parameter objects -- and therefore their Adam optimizer momentum -- untouched.

Run with:
    python test_inplace_resize.py
"""
import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model

MODEL_NAME = "google/vit-base-patch16-224"
NUM_CLASSES = 100

layer_ranks = {i: 16 for i in range(12)}

base = ViTForImageClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
rank_pattern = {}
for i, r in layer_ranks.items():
    rank_pattern[f"vit.encoder.layer.{i}.attention.attention.query"] = r
    rank_pattern[f"vit.encoder.layer.{i}.attention.attention.value"] = r
cfg = LoraConfig(r=16, rank_pattern=rank_pattern, lora_alpha=16, lora_dropout=0.1,
                  target_modules=["query", "value"], bias="none")
model = get_peft_model(base, cfg)

def get_lora_module(model, layer_idx, proj):
    target_suffix = f"layer.{layer_idx}.attention.attention.{proj}"
    for name, module in model.named_modules():
        if name.endswith(target_suffix) and hasattr(module, "lora_A"):
            return module
    return None

# Sanity check the module actually has the attributes we expect
mod0 = get_lora_module(model, 0, "query")
print("Found module for layer 0 query:", mod0 is not None)
if mod0 is not None:
    print("  has lora_A:", hasattr(mod0, "lora_A"), " has lora_B:", hasattr(mod0, "lora_B"))
    print("  has r:", hasattr(mod0, "r"), " has scaling:", hasattr(mod0, "scaling"))
    print("  lora_A['default'] shape:", mod0.lora_A["default"].weight.shape)
    print("  lora_B['default'] shape:", mod0.lora_B["default"].weight.shape)
    print("  current r['default']:", mod0.r["default"])

# Capture parameter object identities for ALL layers BEFORE resizing layer 5
before_ids = {}
for name, p in model.named_parameters():
    if p.requires_grad:
        before_ids[name] = id(p)

def resize_layer_inplace(model, layer_idx, new_r, lora_alpha=16):
    for proj in ["query", "value"]:
        module = get_lora_module(model, layer_idx, proj)
        old_A = module.lora_A["default"].weight.data.clone()
        old_B = module.lora_B["default"].weight.data.clone()
        old_r = old_A.shape[0]
        in_features, out_features = old_A.shape[1], old_B.shape[0]
        if new_r == old_r:
            continue
        elif new_r > old_r:
            new_A = torch.zeros(new_r, in_features); new_A[:old_r] = old_A
            new_B = torch.zeros(out_features, new_r); new_B[:, :old_r] = old_B
        else:
            W = old_B @ old_A
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            sqrt_S = torch.sqrt(S[:new_r].clamp(min=0.0))
            new_B = U[:, :new_r] * sqrt_S
            new_A = Vh[:new_r] * sqrt_S.unsqueeze(1)
        module.lora_A["default"] = nn.Linear(in_features, new_r, bias=False)
        module.lora_A["default"].weight = nn.Parameter(new_A)
        module.lora_B["default"] = nn.Linear(new_r, out_features, bias=False)
        module.lora_B["default"].weight = nn.Parameter(new_B)
        module.r["default"] = new_r
        module.scaling["default"] = lora_alpha / new_r
    return model

model = resize_layer_inplace(model, layer_idx=5, new_r=64)

# Now check: layer 5's params should be NEW objects (different id / different shape).
# All other layers' params should be UNCHANGED (same id as before).
after_ids = {}
for name, p in model.named_parameters():
    if p.requires_grad:
        after_ids[name] = id(p)

changed_outside_layer5 = []
layer5_changed = False
for name in before_ids:
    if name not in after_ids:
        continue
    if "layer.5." in name:
        if before_ids[name] != after_ids[name]:
            layer5_changed = True
    else:
        if before_ids[name] != after_ids[name]:
            changed_outside_layer5.append(name)

layer5_module = get_lora_module(model, 5, "query")
print(f"\\nLayer 5 new rank: {layer5_module.r['default']} (expected 64)")
print(f"Layer 5 params changed identity: {layer5_changed} (expected True)")
print(f"Params OUTSIDE layer 5 that changed identity: {len(changed_outside_layer5)} (expected 0)")

if layer5_changed and len(changed_outside_layer5) == 0 and layer5_module.r["default"] == 64:
    print("\\nPASS -- in-place resize works correctly. Optimizer state for all other")
    print("layers would be fully preserved. Safe to proceed with the full patch.")
else:
    print("\\nFAIL -- something is off. Send me this full output before proceeding.")
