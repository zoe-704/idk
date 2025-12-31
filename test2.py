import torch
import time
from transformer_lens import HookedTransformer

# device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# load model
model = HookedTransformer.from_pretrained("pythia-160m").to(device)
model.eval()

seq_len = 128

print(f"Running on device: {device}")
print("bs | time (ms) | tokens/sec")
print("-" * 30)

for bs in [1, 4, 8, 16, 32]:
    tokens = torch.randint(
        0, model.cfg.d_vocab,
        (bs, seq_len),
        device=device
    )

    # warmup (important for MPS)
    with torch.no_grad():
        model(tokens)

    torch.mps.synchronize() if device == "mps" else None

    start = time.time()
    with torch.no_grad():
        model(tokens)
    torch.mps.synchronize() if device == "mps" else None
    end = time.time()

    elapsed = end - start
    tokens_processed = bs * seq_len
    tps = tokens_processed / elapsed

    print(f"{bs:2d} | {elapsed*1000:8.2f} | {tps:10.1f}")

