"""
scripts/evaluate_models.py

Evaluate multiple cover-letter generation systems (baselines + RAG+LoRA)
on the processed test set, using:

- ROUGE-L (generated vs reference cover letter)
- Job relevance (cosine similarity between generated letter and job description)
- Resume alignment (cosine similarity between generated letter and resume)

Outputs:
- experiments/results/metrics.json               (system-level metrics, all systems)
- experiments/results/ablation_results.json      (subset for RAG/LoRA ablation)
- experiments/results/qualitative_examples.json  (sampled examples for error analysis)
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import evaluate  # pip install evaluate

#paths
PROC_DIR = Path("data/processed")
RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = RESULTS_DIR / "metrics.json"
ABLATION_PATH = RESULTS_DIR / "ablation_results.json"
QUAL_EXAMPLES_PATH = RESULTS_DIR / "qualitative_examples.json"

#importing generation
from src.baselines import (
    generate_template_baseline,
    generate_prompt_only_baseline,
)
from src.inference import (
    generate_rag_lora_model,
    generate_lora_only_model,   # LoRA without RAG 
    generate_rag_only_model,    # RAG without LoRA 
)


#Loading data
def load_test_data(path: Path) -> List[Dict]:
    """Load JSONL test set produced by build_original_dataset.py"""
    records = []
    with path.open() as f:
        for line in f:
            records.append(json.loads(line))
    return records


def parse_job_and_resume(input_text: str) -> Tuple[str, str]:
    """
    Given the instruction-style `input` string produced by build_original_dataset.py,
    recover the `job_text` and `resume_text`.

    Expected pattern (from build_instruction_example):

        "JOB DESCRIPTION:\\n"
        {job_text}
        "\\n\\nRESUME:\\n"
        {resume_text}
        "\\n\\nWrite a 3–5 paragraph cover letter..."

    We split on these markers robustly.
    """
    job_text = ""
    resume_text = ""

    try:
        #Split after "JOB DESCRIPTION:\n"
        after_job = input_text.split("JOB DESCRIPTION:\n", 1)[1]

        #job_text ends right before "\n\nRESUME:\n"
        job_text_part, after_resume_marker = after_job.split("\n\nRESUME:\n", 1)
        job_text = job_text_part.strip()

        #resume_text ends before the next "\n\nWrite" (instructions)
        if "\n\nWrite" in after_resume_marker:
            resume_text_part, _ = after_resume_marker.split("\n\nWrite", 1)
        else:
            #Fallback if format slightly differs
            resume_text_part = after_resume_marker
        resume_text = resume_text_part.strip()
    except Exception:
        #If parsing fails for any reason, just return empty strings;
        #downstream metrics will treat them as low-similarity.
        job_text = ""
        resume_text = ""

    return job_text, resume_text


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def evaluate_system(
    name: str,
    generator_fn: Callable[[str, str], str],
    records: List[Dict],
    rouge_metric,
    embed_model,
) -> Dict:
    """
    Evaluate a single system on:
    - ROUGE-L (generated vs reference cover letter)
    - Job relevance (sim(gen, job_text))
    - Resume alignment (sim(gen, resume_text))

    `records` come from the new processed format:
      ex["input"]  = full instruction prompt
      ex["output"] = reference cover letter
    """

    all_rouge_preds: List[str] = []
    all_rouge_refs: List[str] = []

    job_relevance_scores: List[float] = []
    resume_alignment_scores: List[float] = []

    for ex in tqdm(records, desc=f"Evaluating {name}"):
        input_prompt = ex["input"]
        ref_letter = ex["output"]

        job_text, resume_text = parse_job_and_resume(input_prompt)

        #Generate with the given system (API expects resume_text, job_text)
        gen_letter = generator_fn(resume_text, job_text)

        #ROUGE-L
        all_rouge_preds.append(gen_letter)
        all_rouge_refs.append(ref_letter)

        #Job relevance: sim(gen, job_text)
        emb_gen = embed_model.encode(gen_letter, convert_to_numpy=True)
        emb_job = embed_model.encode(job_text, convert_to_numpy=True)
        job_relevance_scores.append(cosine_sim(emb_gen, emb_job))

        #Resume alignment: sim(gen, resume_text)
        emb_resume = embed_model.encode(resume_text, convert_to_numpy=True)
        resume_alignment_scores.append(cosine_sim(emb_gen, emb_resume))

    #Compute ROUGE-L
    rouge_result = rouge_metric.compute(
        predictions=all_rouge_preds,
        references=all_rouge_refs,
        rouge_types=["rougeL"],
        use_stemmer=True,
    )
    rougeL_score = float(rouge_result["rougeL"])

    metrics = {
        "rougeL": rougeL_score,
        "job_relevance_mean": float(np.mean(job_relevance_scores)),
        "resume_alignment_mean": float(np.mean(resume_alignment_scores)),
    }

    return metrics


def collect_qualitative_examples(
    system_name: str,
    generator_fn: Callable[[str, str], str],
    records: List[Dict],
    num_examples: int = 5,
) -> List[Dict]:
    """
    Sample a few examples from the test set and store:
    - system name
    - resume_text
    - job_text
    - reference (ground truth cover letter)
    - generated (model output)
    """
    import random

    if len(records) == 0:
        return []

    sampled = random.sample(records, k=min(num_examples, len(records)))
    examples: List[Dict] = []

    for ex in sampled:
        input_prompt = ex["input"]
        ref_letter = ex["output"]

        job_text, resume_text = parse_job_and_resume(input_prompt)
        gen_letter = generator_fn(resume_text, job_text)

        examples.append({
            "system": system_name,
            "resume_text": resume_text,
            "job_text": job_text,
            "reference": ref_letter,
            "generated": gen_letter,
            "notes": "",
        })

    return examples


def main():
    #fast mode
    FAST_MODE = True             
    MAX_EXAMPLES = 50            

    #Loading data
    test_path = PROC_DIR / "test.jsonl"
    records = load_test_data(test_path)
    print(f"Loaded {len(records)} test examples from {test_path}")

    if FAST_MODE:
        records = records[:MAX_EXAMPLES]
        print(f"[FAST MODE] Using only first {len(records)} examples")

    #Loading metrics + embedding tools
    rouge_metric = evaluate.load("rouge")
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    #Defining systems we might evaluate
    systems: Dict[str, Callable[[str, str], str]] = {
        "template_baseline": generate_template_baseline,
        "prompt_only_baseline": generate_prompt_only_baseline,
        "rag_lora": generate_rag_lora_model,
    }

    #Adding ablation variants if imported properly
    if generate_lora_only_model is not None:
        systems["lora_only"] = generate_lora_only_model
    if generate_rag_only_model is not None:
        systems["rag_only"] = generate_rag_only_model

    #Loading existing metrics (so we skip re-running them)
    existing_results: Dict[str, Dict] = {}
    if METRICS_PATH.exists():
        with METRICS_PATH.open() as f:
            existing_results = json.load(f)
        print(f"Found existing metrics for systems: {list(existing_results.keys())}")

    all_results: Dict[str, Dict] = dict(existing_results)

    #Running only missing systems
    for name, fn in systems.items():
        if name in existing_results:
            print(f"Skipping {name} (already in metrics.json)")
            continue

        print(f"\n=== Evaluating system: {name} ===")
        metrics = evaluate_system(name, fn, records, rouge_metric, embed_model)
        print(f"Results for {name}: {metrics}")
        all_results[name] = metrics

    #Saving merged metrics
    with METRICS_PATH.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved merged system metrics to {METRICS_PATH}")

    #Building ablation results (only 4 systems)
    ablation_candidates = ["prompt_only_baseline", "lora_only", "rag_only", "rag_lora"]
    ablation_results = {
        name: all_results[name]
        for name in ablation_candidates
        if name in all_results
    }

    with ABLATION_PATH.open("w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"Saved ablation results to {ABLATION_PATH}")

    #Qualitative examples 
    if "rag_lora" in systems:
        qual_examples = collect_qualitative_examples(
            system_name="rag_lora",
            generator_fn=systems["rag_lora"],
            records=records,
            num_examples=5,
        )

        with QUAL_EXAMPLES_PATH.open("w") as f:
            json.dump(qual_examples, f, indent=2)

        print(f"Saved qualitative examples to {QUAL_EXAMPLES_PATH}")
    else:
        print("rag_lora missing — skipping qualitative examples.")

if __name__ == "__main__":
    main()