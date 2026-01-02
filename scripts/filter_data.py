from load_dataset import load_stackexchange

# === CONFIG ===
MIN_LENGTH = 50    # Minimum words per example
MAX_LENGTH = 512   # Maximum words per example
MAX_EXAMPLES = 50_000  # Optional: limit number of examples processed

def length_filter(example, min_length=MIN_LENGTH, max_length=MAX_LENGTH):
    """Return True if example text is within length bounds."""
    length = len(example['text'].split())  # count words
    return min_length <= length <= max_length

def filter_dataset(dataset, min_length=MIN_LENGTH, max_length=MAX_LENGTH, max_examples=None):
    """Generator that yields only examples passing the length filter."""
    count = 0
    for example in dataset:
        if length_filter(example, min_length, max_length):
            yield example
            count += 1
        if max_examples is not None and count >= max_examples:
            break

if __name__ == "__main__":
    # Load raw dataset
    raw_ds = load_stackexchange(streaming=True)
    print("Raw dataset loaded (streaming)")

    # Apply length filter
    filtered_gen = filter_dataset(raw_ds)
    
    # Collect filtered examples into a list (optional: could also save directly to disk)
    filtered_examples = list(filtered_gen)
    print(f"Filtered dataset size: {len(filtered_examples)}")

    # Peek at a sample
    if filtered_examples:
        print("\nSample filtered text:\n")
        print(filtered_examples[0]["text"][:500])

