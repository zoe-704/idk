import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

# device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
model = HookedTransformer.from_pretrained("pythia-160m").to(device)
model.eval()

text = ["Hello world", "This is a test", "TransformerLens is awesome"]
tokens = tokenizer(text, return_tensors="pt", padding=True).input_ids.to(device)

def cache_hook(act, hook):
    return act  # simply stores the activation

logits, cache = model.run_with_cache(
    tokens,
    names_filter=lambda name: "attn" in name or "mlp" in name
)

# Print the first 5 cached activations
for k in list(cache.keys())[:5]:
    print(k, cache[k].shape)

for k, v in cache.items():
    print(k, v.shape)

with torch.no_grad():
    logits, cache = model.run_with_cache(
        tokens,
        names_filter=lambda name: "attn" in name or "mlp" in name
    )

