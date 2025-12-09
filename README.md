# CoverLetter-RAG: Retrieval-Augmented Cover Letter Generation with LoRA-Fine-Tuned T5
Live website: <https://andy157ko-coverletter-rag-app-streamlit-3e1rik.streamlit.app/>
A retrieval-augmented machine learning system that generates personalized cover letters using a user’s resume and job description. The system fine-tunes a T5 model with LoRA, retrieves semantically similar examples with FAISS, and constructs a structured RAG prompt for high-quality, grounded cover letter generation.

---

## What it Does

This project implements an end-to-end Retrieval-Augmented Generation pipeline for personalized cover letter generation. The system:

- Builds a dataset of resume/job/cover-letter triples  
- Preprocesses text and constructs embeddings  
- Builds a FAISS vector index for semantic retrieval  
- Fine-tunes a T5 model using LoRA for efficient SFT  
- Generates cover letters using retrieved examples + structured prompting  
- Includes baselines and full evaluation pipeline  
- Provides a CLI demo that takes resume + job text and outputs a full cover letter  

---

## Quick Start

### 1. Installation

```
git clone <your-repo-url>
cd coverletter-rag

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Build Embeddings + FAISS Index (Do Not do step 2,3,4 without data! If you just want to run the model with my currently trained model skip to step 5)
```
python scripts/build_embeddings.py
--data_path data/processed/train.jsonl
--output_dir data/embeddings
```

### 3. Train LoRA Model
```
python -m src.train --config_name rag_lora
```
Artifacts saved:
- runs/rag_lora/best_model.pt  
- TensorBoard logs in runs/

### 4. Evaluate All Models
```
python scripts/evaluate_models.py
```
Outputs saved to:
- experiments/results/metrics.json  
- experiments/results/qualitative_examples.json  

### 5. Run Demo
```
python scripts/run_demo_cli.py
--resume_path examples/my_resume.txt
--job_path examples/my_job.txt
```
To see web-based version
```
setup_env.sh
run_streamlit.sh
```
---

## Video Links
- Demo Video: <https://drive.google.com/file/d/1Ux7b5Vt6IZnc6g5OSNxsyI0djLnGnNFj/view?usp=sharing>
- Technical Walkthrough Video: <https://drive.google.com/file/d/1YnypbG4ztzQ_Nxn2hVEaa-IdwafwYbsr/view?usp=sharing>  
---

### Metrics Used
- ROUGE-L (generation quality)
- Cosine similarity between:
  - generated ↔ job description
  - generated ↔ resume
  - generated ↔ reference letter

### Baseline Models

| System | Description |
|--------|-------------|
| Template Baseline | Handcrafted template output |
| Prompt-Only Baseline | Generation with no retrieval + no fine-tuning |
| RAG + LoRA (Final Model) | Full retrieval pipeline + LoRA tuned generator |

###  Results

| System | ROUGE-L | Resume Similarity | Job Similarity |
|--------|---------|-------------------|----------------|
| Template | 0.32 | 0.79 | 0.74 |
| Prompt-Only | 0.00 | 0.11 | 0.07 |
| RAG + LoRA | 0.27 | 0.59 | 0.53 |

### Qualitative Observations
- Retrieval provides strong grounding in resume content  
- LoRA fine-tuning significantly improves personalization  
- Baselines lack specificity and correctness  
- Rule-based fallback ensures minimum-quality output  

---

## Individual Contributions

This project was completed individually.  
I contributed:

- Data preprocessing pipeline  
- Embedding + FAISS retrieval system  
- RAG prompt engineering  
- LoRA fine-tuning implementation  
- Full training loop with logging  
- Evaluation pipeline (metrics + qualitative samples)  
- Baseline systems  
- CLI demo  
- All documentation + videos  
