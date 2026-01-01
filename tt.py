import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

# 1. Setup Device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# 2. Load Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
# FIX: Define the padding token
tokenizer.pad_token = tokenizer.eos_token 

model = HookedTransformer.from_pretrained("pythia-160m").to(device)
model.eval()

# 3. Prepare Data
text = ["Hello world", "This is a test", "TransformerLens is awesome"]
# Added padding=True and truncation=True for safety at scale
tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

# 4. Run with Cache (Capturing Attn, MLP, and Residuals)
# We use a lambda to filter for all three components you mentioned
with torch.no_grad():
    logits, cache = model.run_with_cache(
        tokens,
        names_filter=lambda name: any(x in name for x in ["attn", "mlp", "hook_resid"])
    )

# 5. Print Results
print(f"Successfully cached {len(cache)} activation points.")

# Show a sample of different types
for k in list(cache.keys())[:10]:
    print(f"{k:<30} | Shape: {cache[k].shape}")
