# Stack Exchange bc Wikipedia contains several languuages
from datasets import load_dataset

def load_stackexchange(streaming=True, max_examples=50_000):
    ds = load_dataset(
        "togethercomputer/RedPajama-Data-1T",
        "stackexchange",
        split="train",
        streaming=streaming,
    )

    if max_examples is not None:
        ds = ds.take(max_examples)

    return ds


if __name__ == "__main__":
    ds = load_stackexchange()

    ex = next(iter(ds))
    print("Keys:", ex.keys())
    print("\nSample text:\n")
    print(ex["text"][:500])

