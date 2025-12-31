import torch
from transformer_lens import HookedTransformer

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = HookedTransformer.from_pretrained("pythia-160m").to(device)


