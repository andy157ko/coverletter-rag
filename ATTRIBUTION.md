# ATTRIBUTION

## Datasets

- **Cover Letter Dataset** by ShashiVish (HuggingFace: [`ShashiVish/cover-letter-dataset`](https://huggingface.co/datasets/ShashiVish/cover-letter-dataset))
  - Used as the primary training and evaluation dataset
  - Contains resume, job description, and cover letter triplets
  - Split into train/validation/test sets (85/15 split for train/val)

## Models

### Base Language Models

- **Flan-T5-Base** (Google) - HuggingFace: [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base)
  - Used as the base sequence-to-sequence model
  - Fine-tuned using LoRA (Low-Rank Adaptation) in this project
  - LoRA configuration: rank=16, alpha=32, dropout=0.05

- **T5-Base** (Google) - HuggingFace: [`t5-base`](https://huggingface.co/t5-base)
  - Used for the prompt-only baseline (no fine-tuning, no RAG)
  - Provides comparison against the fine-tuned system

### Embedding Models

- **all-MiniLM-L6-v2** (SentenceTransformers) - HuggingFace: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  - Used for generating embeddings for FAISS-based retrieval
  - Creates semantic representations of resume/job/cover letter text

## Tools and Libraries

### Core ML/AI Frameworks
- **PyTorch** - Deep learning framework for model training and inference
- **Transformers** (HuggingFace) - Pre-trained model loading and tokenization
- **PEFT** (Parameter-Efficient Fine-Tuning) - LoRA implementation
- **Accelerate** - Distributed training and inference acceleration

### Data Processing
- **Datasets** (HuggingFace) - Dataset loading and management
- **scikit-learn** - Train/validation/test splitting
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations

### Retrieval and Embeddings
- **FAISS** (Facebook AI Similarity Search) - Efficient vector similarity search for RAG retrieval
- **sentence-transformers** - Sentence embedding models

### Evaluation
- **evaluate** (HuggingFace) - Metrics computation (ROUGE-L)
- **rouge_score** - ROUGE metric implementation
- **nltk** - Natural language processing utilities

### Development and Deployment
- **Streamlit** - Web application framework for the demo interface
- **TensorBoard** - Training visualization and logging
- **tqdm** - Progress bars for long-running operations

### Utilities
- **tiktoken** - Fast tokenization
- **sentencepiece** - Subword tokenization
- **python-dotenv** - Environment variable management

## AI Assistance

- **ChatGPT (OpenAI)** was used to help:
  - Design the project folder structure and initial code skeleton
  - Draft parts of this documentation and some boilerplate code
  - Debug and troubleshoot implementation issues
  - **Cursor** was used to help:
  - Finishg & Deploy the frontend of the website
- All final code, experiments, and evaluations were reviewed and adapted by the author.

## Other Sources

### Papers and Research
- **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021)
  - Inspiration for parameter-efficient fine-tuning approach
- **Retrieval-Augmented Generation (RAG)** concepts from various NLP research
  - Applied to cover letter generation domain

### Documentation and Tutorials
- HuggingFace Transformers documentation for model fine-tuning
- FAISS documentation for vector search implementation
- Streamlit documentation for web app development
