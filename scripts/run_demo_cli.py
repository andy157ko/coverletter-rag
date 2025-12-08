"""
Simple CLI demo script for your non-technical demo video.

Usage:
    python scripts/run_demo_cli.py \
        --resume_path examples/my_resume.txt \
        --job_path examples/my_job.txt
"""

import argparse
from pathlib import Path
import sys

# --- Ensure repo root (which contains `src/`) is on sys.path ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import generate_rag_lora_model


def load_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return p.read_text()


def main(args):
    # 1. Load raw text inputs
    resume_text = load_text(args.resume_path)
    job_text = load_text(args.job_path)

    # 2. Call the full RAG + LoRA system
    cover_letter = generate_rag_lora_model(resume_text, job_text)

    # 3. Print result
    print("===== GENERATED COVER LETTER =====")
    print()
    print(cover_letter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a cover letter from a resume + job description."
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        required=True,
        help="Path to a plain-text resume file",
    )
    parser.add_argument(
        "--job_path",
        type=str,
        required=True,
        help="Path to a plain-text job description file",
    )
    args = parser.parse_args()
    main(args)
