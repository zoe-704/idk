import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

# devices
gpu_device = "mps" if torch.backends.mps.is_available() else "cpu"
cpu_device = "cpu"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")

# load models
model_gpu = HookedTransformer.from_pretrained("pythia-160m").to(gpu_device)
model_cpu = HookedTransformer.from_pretrained("pythia-160m").to(cpu_device)

# tokens
tokens_gpu = tokenizer("Hello world", return_tensors="pt").input_ids.to(gpu_device)
tokens_cpu = tokenizer("Hello world", return_tensors="pt").input_ids.to(cpu_device)

# forward passes
with torch.no_grad():
    gpu_logits = model_gpu(tokens_gpu)
    cpu_logits = model_cpu(tokens_cpu)

# compare
print(torch.allclose(cpu_logits, gpu_logits.cpu(), atol=1e-4))

