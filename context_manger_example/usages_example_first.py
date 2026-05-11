from transformers import AutoModelForCausalLM, AutoTokenizer
from example_first import GPUMemoryManager
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").cuda()

prompt = "Explain async programming"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    with GPUMemoryManager(mixed_precision=True):
        outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0]))