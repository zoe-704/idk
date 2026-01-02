import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "EleutherAI/pythia-160m"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.eval()

print("Model loaded")
print("Number of parameters:", sum(p.numel() for p in model.parameters()))

text = "Hello, this is a test sentence."
inputs = tokenizer(text, return_tensors="pt")  # CPU by default

with torch.no_grad():
    outputs = model(**inputs)

print("Logits shape:", outputs.logits.shape)

