# scripts/build_original_dataset.py

import json
from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")

# -------------------------------------------------------------
# BUILD JOB + RESUME TEXT FIELDS AS BEFORE
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# CONVERT INTO INSTRUCTION-TUNING FORMAT
# -------------------------------------------------------------
def build_instruction_example(job_text, resume_text, cover_letter_text):
    """
    Convert each dataset item into:
      { "input": "...", "output": "..." }
    """

    prompt = (
        "You are an expert career coach who writes tailored, polished, multi-paragraph cover letters.\n"
        "Your task is to write a NEW, professional, personalized cover letter based ONLY on the\n"
        "job description and resume provided below.\n\n"

        "JOB DESCRIPTION:\n"
        f"{job_text}\n\n"

        "RESUME:\n"
        f"{resume_text}\n\n"

        "Write a 3–5 paragraph cover letter that:\n"
        "- Clearly connects resume achievements to job qualifications.\n"
        "- Avoids generic filler phrases.\n"
        "- Stays factual (no added skills or experience).\n"
        "- Uses polished, professional language.\n\n"
        "Begin your cover letter now:\n"
    )

    return {
        "input": prompt,
        "output": cover_letter_text.strip()
    }

# -------------------------------------------------------------
# JSONL WRITER
# -------------------------------------------------------------
def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

# -------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------
def main():
    print("Loading HF dataset...")
    ds = load_dataset("ShashiVish/cover-letter-dataset")
    train_ds = ds["train"]
    test_ds = ds["test"]

    print("Building train records...")
    train_records = []
    for ex in train_ds:
        job_text, resume_text = build_text_fields(ex)
        formatted = build_instruction_example(job_text, resume_text, ex["Cover Letter"])
        train_records.append(formatted)

    print("Building test records...")
    test_records = []
    for ex in test_ds:
        job_text, resume_text = build_text_fields(ex)
        formatted = build_instruction_example(job_text, resume_text, ex["Cover Letter"])
        test_records.append(formatted)

    # Split train into train/val sets
    print("Splitting train/val...")
    train_split, val_split = train_test_split(
        train_records, test_size=0.15, random_state=42
    )

    # Write outputs
    print("Writing processed datasets...")
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(PROC_DIR / "train.jsonl", train_split)
    write_jsonl(PROC_DIR / "val.jsonl", val_split)
    write_jsonl(PROC_DIR / "test.jsonl", test_records)

    print("Done! Instruction-tuning datasets written to data/processed.")

if __name__ == "__main__":
    main()
