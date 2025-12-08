# CoverLetter-RAG: Retrieval-Augmented Cover Letter Generation with LoRA-Fine-Tuned T5

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
