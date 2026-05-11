import torch
from transformers import AutoModelForCausalLM

def profile_model(model_name):
    print(f"--- Profiling {model_name} ---")
    
    # Check if a GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device detected: {device.upper()}")

    precisions = [
        {"name": "FP32", "dtype": torch.float32, "quant": False}, # Full precision
        {"name": "FP16", "dtype": torch.float16, "quant": False}, # Half precision (note: on CPU this won't actually reduce memory, but we can still test loading)
        {"name": "INT8", "dtype": None, "quant": True} # 8-bit quantization (requires bitsandbytes and a compatible model)
    ]

    for p in precisions:
        try:
            print(f"\nLoading in {p['name']}...")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name, # A small model to ensure it runs on 16GB RAM
                torch_dtype=p['dtype'], # Use specified dtype for loading
                load_in_8bit=p['quant'], # Enable 8-bit quantization if specified
                device_map="auto" # Automatically handles memory placement
            )
            
            # Use HuggingFace's built-in footprint tool
            usage_gb = model.get_memory_footprint() / (1024**3) # Convert bytes to GB
            print(f"Done! Memory used: {usage_gb:.2f} GB")
            
            # Cleanup to avoid OOM for the next precision test
            del model
            if device == "cuda": # Clear GPU memory after each test
                torch.cuda.empty_cache() # Clear GPU memory
                
        except Exception as e:
            print(f"Could not load {p['name']}: {e}")

if __name__ == "__main__":
    # Try a small model first to test the script
    profile_model("gpt2")  # A small model to ensure it runs on 16GB RAM
