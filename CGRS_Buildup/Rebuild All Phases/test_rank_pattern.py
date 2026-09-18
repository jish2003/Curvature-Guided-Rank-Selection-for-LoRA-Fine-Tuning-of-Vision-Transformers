"""
Quick diagnostic — verifies PEFT's rank_pattern actually produces
heterogeneous per-layer LoRA ranks, BEFORE running any real training.

Run with:
    python test_rank_pattern.py

Expected output: "MATCH" for the parameter count check. If it prints
"MISMATCH", the rank_pattern keys need adjusting before touching the
main training script at all.
"""
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model

MODEL_NAME = "google/vit-base-patch16-224"
NUM_CLASSES = 100

# Deliberately uneven ranks to make any bug obvious
test_ranks = {(i, proj): (64 if i >= 9 else 16) for i in range(12) for proj in ["query", "value"]}

def expected_params(ranks):
    # For a 768-hidden-size Linear layer: A is (r, 768), B is (768, r) -> 1536*r params per adapter
    return sum(1536 * r for r in ranks.values())

def build_pattern_keys(ranks):
    return {f"vit.encoder.layer.{i}.attention.attention.{proj}": r
            for (i, proj), r in ranks.items()}

def build_model(ranks, default_r=16):
    base = ViTForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)

    # Print actual module names once, so we can visually confirm the naming
    # convention matches what we assume below.
    print("Sample matching module names found in base model (layer 0, query/value):")
    for name, _ in base.named_modules():
        if "layer.0.attention.attention.query" in name or "layer.0.attention.attention.value" in name:
            print(f"    {name}")

    cfg = LoraConfig(
        r=default_r,
        rank_pattern=build_pattern_keys(ranks),
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        bias="none",
    )
    return get_peft_model(base, cfg)

model = build_model(test_ranks)
actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
expected = expected_params(test_ranks)

print(f"\nExpected trainable params (heterogeneous ranks): {expected:,}")
print(f"Actual trainable params from model:               {actual:,}")

if actual == expected:
    print("\nMATCH — rank_pattern is working correctly. Safe to proceed with full rebuild.")
else:
    print("\nMISMATCH — rank_pattern keys are not matching PEFT's internal module names.")
    print("Do NOT proceed with the full rebuild yet. Send me this output (including the")
    print("'Sample matching module names' printed above) so I can fix the key format.")
