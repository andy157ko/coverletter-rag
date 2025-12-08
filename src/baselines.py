# src/baselines.py

"""
Baselines for cover letter generation.

Exposed functions (used by scripts/evaluate_models.py):
- generate_template_baseline(resume_text, job_text) -> str
- generate_prompt_only_baseline(resume_text, job_text) -> str
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# -----------------------------------------------------------------------------
# Simple template baseline (no ML)
# -----------------------------------------------------------------------------

def generate_template_baseline(resume_text: str, job_text: str) -> str:
    """
    Very simple baseline that ignores most structure and just plugs the
    resume and job into a fixed template. This is intentionally weak so
    our ML systems have something to beat.
    """

    # Very naive heuristic: truncate resume & job so it's not enormous
    short_resume = resume_text[:600]
    short_job = job_text[:400]

    letter = (
        "Dear Hiring Manager,\n\n"
        "I am writing to express my interest in the position described below.\n\n"
        "Job Description (excerpt):\n"
        f"{short_job}\n\n"
        "From my background, I believe I am a strong fit. For example:\n"
        f"{short_resume}\n\n"
        "I would welcome the opportunity to contribute to your team and learn more about this role.\n\n"
        "Sincerely,\n"
        "[Your Name]\n"
    )

    return letter


# -----------------------------------------------------------------------------
# Prompt-only baseline using a generic T5 model (no RAG, no LoRA)
# -----------------------------------------------------------------------------

# You can change this to another general model if you want.
_BASELINE_MODEL_NAME = "t5-base"

# Optional HF token if you need it for private models (not required for t5-base)
_HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)

# Load once at import so we don't reload per example
_baseline_tokenizer = AutoTokenizer.from_pretrained(
    _BASELINE_MODEL_NAME,
    use_auth_token=_HF_TOKEN if _HF_TOKEN else None,
)

_baseline_model = AutoModelForSeq2SeqLM.from_pretrained(
    _BASELINE_MODEL_NAME,
    use_auth_token=_HF_TOKEN if _HF_TOKEN else None,
)

# For simplicity and to avoid device mismatches, we keep this on CPU.
_device = torch.device("cpu")
_baseline_model.to(_device)
_baseline_model.eval()


def generate_prompt_only_baseline(resume_text: str, job_text: str) -> str:
    """
    Baseline that uses a generic pretrained T5 model with a single prompt,
    without any retrieval or fine-tuning specific to cover letters.
    """

    prompt = (
        "You are an assistant that writes tailored cover letters for job applications.\n\n"
        "Write a professional, concise cover letter using the resume and job description.\n"
        "Do not invent fake degrees or jobs. Use only information actually present in the resume.\n\n"
        f"[JOB]\n{job_text}\n\n"
        f"[RESUME]\n{resume_text}\n\n"
        "Write ONLY the cover letter text:\n"
    )

    inputs = _baseline_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(_device)

    with torch.no_grad():
        outputs = _baseline_model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
        )

    return _baseline_tokenizer.decode(outputs[0], skip_special_tokens=True)
