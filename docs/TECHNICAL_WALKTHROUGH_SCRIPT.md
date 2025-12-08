# Technical Walkthrough Script (5-6 minutes)

## Video Structure & Navigation Guide

---

### [0:00-0:30] Introduction & Project Overview
**Files to show:**
- `README.md` (lines 1-20)

**Script:**
"Hi, I'm [your name], and today I'll walk you through my Cover Letter RAG project. This is an end-to-end machine learning system that generates personalized cover letters using Retrieval-Augmented Generation with a LoRA fine-tuned T5 model.

The system takes a user's resume and a job description, retrieves semantically similar examples from a training corpus, and generates a tailored cover letter. Let me show you the architecture and implementation."

---

### [0:30-1:30] Project Architecture & Structure
**Files to show:**
- Project root directory structure
- `src/config.py` (show all three dataclasses)

**Script:**
"Let's start with the project structure. I've organized this into clear modules: data processing, model training, RAG pipeline, and evaluation.

Here in `config.py`, I define three configuration classes. The `TrainingConfig` specifies our base model - Flan-T5-base - with LoRA parameters: rank 16, alpha 32, and dropout 0.05. The `DataConfig` points to our train/val/test splits, and `RAGConfig` specifies the embedding model and FAISS index path.

This modular design makes it easy to experiment with different hyperparameters and configurations."

---

### [1:30-2:30] Data Pipeline & Preprocessing
**Files to show:**
- `scripts/build_original_dataset.py` (show main function, lines 33-68)
- `data/processed/train.jsonl` (show 1-2 example lines)
- `scripts/build_embeddings.py` (entire file)

**Script:**
"Now let's look at the data pipeline. The `build_original_dataset.py` script downloads the HuggingFace cover letter dataset, extracts job descriptions, resumes, and cover letters, then splits them into train, validation, and test sets.

Each example is stored as JSONL with three fields: job_text, resume_text, and cover_letter_text. This format makes it easy to load and process.

Next, `build_embeddings.py` creates our FAISS index for retrieval. It concatenates job, resume, and cover letter text, embeds them using SentenceTransformer's all-MiniLM-L6-v2 model, and builds a FAISS index for fast similarity search. This index allows us to retrieve the top-k most similar examples during inference."

---

### [2:30-3:30] Model Architecture & Training
**Files to show:**
- `src/models.py` (entire file)
- `src/train.py` (show key sections: lines 29-87, 90-140, 142-180)

**Script:**
"The model architecture is straightforward but effective. In `models.py`, I load the base Flan-T5 model and apply LoRA - Low-Rank Adaptation - which allows efficient fine-tuning by only training a small number of additional parameters.

The training script in `train.py` implements a full training loop with several best practices. The collate function tokenizes inputs and targets, masking padding tokens appropriately. I use AdamW optimizer with weight decay for regularization, a linear learning rate schedule with warmup, and gradient clipping to prevent exploding gradients.

The training loop tracks both training and validation loss, implements early stopping with patience, and saves the best model based on validation performance. All metrics are logged to TensorBoard for visualization."

---

### [3:30-4:30] RAG Pipeline & Inference
**Files to show:**
- `src/rag_pipeline.py` (show key functions: lines 20-53, 56-117, 264-296, 298-328, 331-410)
- `src/inference.py` (show _init_rag_pipeline function, lines 67-128)

**Script:**
"The RAG pipeline is where the magic happens. In `rag_pipeline.py`, I have several key components.

First, `extract_title_and_company` uses heuristics to extract job title and company from the job description. `extract_resume_bullets` intelligently extracts bullet points from the resume, filtering out contact information and focusing on experience sections.

The `RAGPipeline` class handles retrieval and generation. The `retrieve` method chunks the resume and job text, embeds all chunks, computes cosine similarity with the query, and returns the top-k most relevant chunks.

The `build_prompt` function is crucial - it constructs a detailed prompt that explicitly tells the model to use the correct job title and company, reference specific resume bullets, and avoid inventing information. This prompt engineering is key to ensuring truthful, personalized output.

Finally, `generate` orchestrates the full pipeline: retrieval, prompt construction, generation, and post-processing. It includes a fallback mechanism that uses a rule-based system if the model output is low quality.

The `inference.py` module handles loading the trained model and FAISS index, wrapping everything in a clean interface."

---

### [4:30-5:00] Evaluation & Results
**Files to show:**
- `scripts/evaluate_models.py` (show evaluate_system function, lines 66-125)
- `experiments/results/metrics.json` (show the results table)

**Script:**
"For evaluation, I implemented a comprehensive metrics suite. The `evaluate_models.py` script evaluates three systems: a template baseline, a prompt-only baseline using T5-base without fine-tuning, and our full RAG+LoRA system.

I measure ROUGE-L for generation quality, and cosine similarity between the generated letter and both the job description and resume to measure relevance and alignment.

As you can see in the results, our RAG+LoRA system outperforms both baselines across all metrics, with a ROUGE-L of 0.324 compared to 0.215 for the template baseline. The resume and job similarity scores are also significantly higher, showing better personalization."

---

### [5:00-5:30] Demo & Web Application
**Files to show:**
- `app_streamlit.py` (show the main structure, lines 1-50, 100-150)
- `scripts/run_demo_cli.py` (entire file)

**Script:**
"Finally, I've built two interfaces for the system. The Streamlit web app provides a user-friendly interface where users can paste their resume and job description, and get a generated cover letter with download functionality.

There's also a CLI demo script that can be run from the command line. Both interfaces use the same inference pipeline, ensuring consistency.

The system is production-ready with proper error handling, loading indicators, and fallback mechanisms to ensure users always get a usable output."

---

### [5:30-6:00] Conclusion & Key Takeaways
**Files to show:**
- Project root (final overview)

**Script:**
"To summarize, this project demonstrates several important ML engineering practices: modular code design, proper train/val/test splits, comprehensive evaluation, and production-ready interfaces.

The key innovations are the combination of RAG for grounding and LoRA for efficient fine-tuning, along with careful prompt engineering to ensure truthful, personalized output. The system outperforms baselines and provides a practical solution for cover letter generation.

Thank you for watching! You can find the full code and documentation in the repository."

---

## Quick Navigation Checklist

**Before recording, have these files open in tabs:**
1. `README.md`
2. `src/config.py`
3. `scripts/build_original_dataset.py`
4. `scripts/build_embeddings.py`
5. `src/models.py`
6. `src/train.py`
7. `src/rag_pipeline.py`
8. `src/inference.py`
9. `scripts/evaluate_models.py`
10. `experiments/results/metrics.json`
11. `app_streamlit.py`

**Tips for recording:**
- Use a code editor with good syntax highlighting (VS Code recommended)
- Zoom in to 120-150% for readability
- Use smooth scrolling between files
- Pause briefly when switching files to let viewers orient
- Highlight specific lines when discussing them (use cursor or selection)

**Timing notes:**
- Total script: ~6 minutes when spoken at normal pace
- Adjust speed based on your speaking pace
- It's okay to go slightly over 6 minutes if needed for clarity

