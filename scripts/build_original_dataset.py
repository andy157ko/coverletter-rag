# scripts/build_original_dataset.py
import json
from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")

def build_text_fields(example):
    job_parts = [
        f"Job Title: {example['Job Title']}",
        f"Hiring Company: {example['Hiring Company']}",
        f"Preferred Qualifications: {example['Preferred Qualifications']}",
    ]
    resume_parts = [
        f"Applicant Name: {example['Applicant Name']}",
        f"Past Experience: {example['Past Working Experience']}",
        f"Current Experience: {example['Current Working Experience']}",
        f"Skillsets: {example['Skillsets']}",
        f"Qualifications: {example['Qualifications']}",
    ]
    job_text = "\n".join(job_parts)
    resume_text = "\n".join(resume_parts)
    return job_text, resume_text

def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def main():
    ds = load_dataset("ShashiVish/cover-letter-dataset")
    train_ds = ds["train"]
    test_ds = ds["test"]

    # Build processed records
    train_records = []
    for ex in train_ds:
        job_text, resume_text = build_text_fields(ex)
        train_records.append({
            "job_text": job_text,
            "resume_text": resume_text,
            "cover_letter_text": ex["Cover Letter"],
        })

    test_records = []
    for ex in test_ds:
        job_text, resume_text = build_text_fields(ex)
        test_records.append({
            "job_text": job_text,
            "resume_text": resume_text,
            "cover_letter_text": ex["Cover Letter"],
        })

    # Split train into train/val
    train_split, val_split = train_test_split(
        train_records, test_size=0.15, random_state=42
    )

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(PROC_DIR / "train.jsonl", train_split)
    write_jsonl(PROC_DIR / "val.jsonl", val_split)
    write_jsonl(PROC_DIR / "test.jsonl", test_records)

if __name__ == "__main__":
    main()

