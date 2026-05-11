import torch
from transformers import AutoModelForCausalLM

def profile_model(model_name):
    print(f"--- Profiling {model_name} ---")
    
    # Check if a GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device detected: {device.upper()}")

    precisions = [
        {"name": "FP32", "dtype": torch.float32, "quant": False},
        {"name": "FP16", "dtype": torch.float16, "quant": False},
        {"name": "INT8", "dtype": None, "quant": True}
    ]

    for p in precisions:
        try:
            print(f"\nLoading in {p['name']}...")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=p['dtype'],
                load_in_8bit=p['quant'],
                device_map="auto" # Automatically handles memory placement
            )
            
            # Use HuggingFace's built-in footprint tool
            usage_gb = model.get_memory_footprint() / (1024**3)
            print(f"Done! Memory used: {usage_gb:.2f} GB")
            
            # Cleanup to avoid OOM for the next precision test
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Could not load {p['name']}: {e}")

if __name__ == "__main__":
    # Try a small model first to test the script
    profile_model("gpt2") 
