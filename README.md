# CoverLetterRAG: Resume-Aware Cover Letter Generator

## What it Does

This project helps students generate tailored cover letters from their resumes and target job descriptions. It uses a retrieval-augmented generation (RAG) pipeline with pretrained transformer language models from HuggingFace, grounding each letter in the student's actual experiences and similar past examples.

## Quick Start

1. Clone the repository and create an environment:

```bash
git clone <YOUR_REPO_URL>.git
cd coverletter-rag
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt