from memory_profiler import profile
from transformers import AutoModel

@profile
def load_bert():
    # Tracks RAM usage as each line executes
    model = AutoModel.from_pretrained("bert-base-uncased") # A small model to ensure it runs on 16GB RAM
    return model

if __name__ == "__main__":
    load_bert() # Run with: python -m memory_profiler profile_me.py
