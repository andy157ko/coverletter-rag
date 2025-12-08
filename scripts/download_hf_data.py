# scripts/download_hf_data.py
from datasets import load_dataset

def main():
    ds = load_dataset("ShashiVish/cover-letter-dataset")
    # This caches under ~/.cache/huggingface by default
    print(ds)

if __name__ == "__main__":
    main()
