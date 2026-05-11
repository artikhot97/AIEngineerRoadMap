import torch
from transformers import AutoModel

def check_mem(label, dtype=None):
    # Load BERT - Safe for 16GB RAM
    model = AutoModel.from_pretrained(
        "bert-base-uncased", 
        torch_dtype=dtype, 
        device_map="auto"
    )
    
    # get_memory_footprint() reports the size of model weights in bytes
    mem_mb = model.get_memory_footprint() / 1e6
    print(f"{label}: {mem_mb:.2f} MB")
    
    del model # Clean up memory for the next precision test

# On 16GB RAM (CPU), these two will run perfectly:
check_mem("FP32 (Full Precision)", dtype=torch.float32)
check_mem("FP16 (Half Precision)", dtype=torch.float16)
