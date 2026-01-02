from load_dataset import load_stackexchange
from datasets import Dataset

# === CONFIG ===
MIN_LENGTH = 50
MAX_LENGTH = 512
MAX_EXAMPLES = 50_000  # maximum filtered examples to keep
SAVE_PATH = "filtered_stackexchange"

# --- Helper functions ---
def length_filter(example, min_length=MIN_LENGTH, max_length=MAX_LENGTH):
    """Return True if example text is within word count bounds."""
    return min_length <= len(example['text'].split()) <= max_length

def filter_dataset(dataset, min_length=MIN_LENGTH, max_length=MAX_LENGTH, max_examples=None):
    """Generator that yields only examples passing the length filter."""
    count = 0
    for example in dataset:
        if length_filter(example, min_length, max_length):
            yield example
            count += 1
        if max_examples is not None and count >= max_examples:
            break

# --- Main ---
if __name__ == "__main__":
    print("Loading raw StackExchange dataset (streaming)...")
    raw_ds = load_stackexchange(streaming=True)
    
    print("Applying length filter...")
    filtered_gen = filter_dataset(raw_ds)

    # Collect filtered examples into a list
    filtered_examples = list(filtered_gen)
    print(f"Filtered dataset size: {len(filtered_examples)}")

    # Convert to Hugging Face Dataset and save
    print(f"Saving filtered dataset to '{SAVE_PATH}'...")
    filtered_ds = Dataset.from_list(filtered_examples)
    filtered_ds.save_to_disk(SAVE_PATH)
    print("Dataset saved successfully!")

    # Inspect a few examples
    print("\nSample filtered example:\n")
    if filtered_examples:
        print(filtered_examples[0]["text"][:500])

