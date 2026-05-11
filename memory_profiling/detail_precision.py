import torch
from transformers import AutoModel

def check_mem(label, dtype=None):
    # Load BERT - Safe for 16GB RAM
    model = AutoModel.from_pretrained(
        "bert-base-uncased",  # A small model to ensure it runs on 16GB RAM
        torch_dtype=dtype,  # Use specified dtype for loading
        device_map="auto" # Automatically handles memory placement
    )
    
    # get_memory_footprint() reports the size of model weights in bytes
    mem_mb = model.get_memory_footprint() / 1e6 # Convert to MB
    print(f"{label}: {mem_mb:.2f} MB")
    
    del model # Clean up memory for the next precision test

# On 16GB RAM (CPU), these two will run perfectly:
check_mem("FP32 (Full Precision)", dtype=torch.float32) # 1.2 GB
check_mem("FP16 (Half Precision)", dtype=torch.float16) # On CPU, this will not actually quantize but we can still check the loading:
