from memory_profiler import profile
from transformers import AutoModel

@profile
def load_bert():
    # Tracks RAM usage as each line executes
    model = AutoModel.from_pretrained("bert-base-uncased")
    return model

if __name__ == "__main__":
    load_bert()
