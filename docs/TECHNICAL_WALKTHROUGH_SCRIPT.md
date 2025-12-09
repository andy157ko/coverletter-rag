# Technical Walkthrough Script (5-6 minutes)

## Video Structure & Navigation Guide

---

### [0:00-0:30] Introduction & Project Overview
**Files to show:**
- `README.md` (lines 1-20, 76-98)

**Script:**
"Hi, I'm [your name], and today I'll walk you through my Cover Letter RAG project. This is an end-to-end machine learning system that generates personalized cover letters using Retrieval-Augmented Generation with a LoRA fine-tuned T5 model.

The system takes a user's resume and a job description, performs semantic retrieval to identify the most relevant qualifications and experiences, and generates a tailored cover letter. As you can see from the results, our RAG+LoRA system achieves a ROUGE-L score of 0.27, with strong job relevance and resume alignment metrics. Let me show you how it all works."

---

### [0:30-1:30] Project Architecture & Configuration
**Files to show:**
- Project root directory structure
- `src/config.py` (entire file - all three dataclasses)

**Script:**
"Let's start with the project structure. I've organized this into clear modules following best practices: source code in `src/`, data processing scripts in `scripts/`, notebooks for analysis, and experiments for logs and results.

Here in `config.py`, I define three configuration classes using Python dataclasses. The `TrainingConfig` specifies our base model - Google's Flan-T5-base - with LoRA parameters: rank 16, alpha 32, and dropout 0.05. We use a learning rate of 5e-5 with weight decay for regularization, and train for 3 epochs with early stopping.

The `DataConfig` points to our train/val/test splits in JSONL format, and `RAGConfig` specifies the embedding model - all-MiniLM-L6-v2 - and FAISS index path. This modular design makes it easy to experiment with different hyperparameters and configurations."

---

### [1:30-2:30] Data Pipeline & Preprocessing
**Files to show:**
- `scripts/build_original_dataset.py` (show main function, lines 76-112, and build_instruction_example, lines 34-62)
- `data/processed/train.jsonl` (show 1-2 example lines to demonstrate input/output format)
- `scripts/build_embeddings.py` (entire file)

**Script:**
"Now let's look at the data pipeline. The `build_original_dataset.py` script downloads the HuggingFace cover letter dataset, extracts structured fields like job title, company, qualifications, and resume experiences.

Crucially, I convert each example into an instruction-tuning format with an `input` field containing a carefully constructed prompt and an `output` field with the reference cover letter. This format is essential for training the model to follow instructions and generate high-quality output.

The script splits the data into train, validation, and test sets using an 85-15 split. Each example is stored as JSONL, making it easy to load and process.

Next, `build_embeddings.py` creates our FAISS index. It concatenates job text, resume text, and the cover letter, embeds them using SentenceTransformer's all-MiniLM-L6-v2 model, and builds a FAISS index for fast similarity search. This index is used during training to potentially retrieve similar examples, though in our final implementation, we use a different retrieval strategy at inference time."

---

### [2:30-3:30] Model Architecture & Training
**Files to show:**
- `src/models.py` (entire file)
- `src/train.py` (show key sections: lines 29-67 collate function, 71-97 setup, 123-180 training loop)

**Script:**
"The model architecture leverages parameter-efficient fine-tuning. In `models.py`, I load the base Flan-T5 model - a sequence-to-sequence transformer - and apply LoRA, Low-Rank Adaptation. LoRA allows us to fine-tune with only 16 rank parameters per layer, making training much more efficient while maintaining performance.

The training script implements a full training loop with several best practices. The collate function handles our instruction-tuning format, tokenizing the input prompts and target cover letters separately, and masking padding tokens appropriately.

I use the AdamW optimizer with weight decay of 0.01 for regularization, a linear learning rate schedule with 100 warmup steps, and gradient clipping with max norm 1.0 to prevent exploding gradients.

The training loop tracks both training and validation loss, implements early stopping with patience of 2 epochs, and saves the best model based on validation performance. All metrics are logged to TensorBoard for visualization. This ensures we get a well-regularized model that generalizes well."

---

### [3:30-4:30] RAG Pipeline & Retrieval Strategy
**Files to show:**
- `src/rag_pipeline.py` (show key functions: lines 17-42 extract_title_and_company, 45-92 extract_resume_bullets, 220-296 retrieve method, 298-390 build_prompt, 393-510 generate method)
- `src/inference.py` (show _init_rag_pipeline, lines 166-232)

**Script:**
"The RAG pipeline is where the system really shines. In `rag_pipeline.py`, I implement a classic RAG retrieval strategy that works over the current user's resume and job description.

First, `extract_title_and_company` uses heuristics to extract the job title and company name from the job description. `extract_resume_bullets` intelligently extracts bullet points from the resume, filtering out contact information and focusing on experience sections.

The `retrieve` method implements our retrieval strategy. Rather than retrieving from a corpus of training examples, we perform retrieval over the current user's own resume and job description. We chunk the resume into bullets and the job description into sentences, embed all chunks, and compute cosine similarity with a query embedding of the combined resume and job text. This identifies the most relevant qualifications from the job and matching experiences from the resume.

The `build_prompt` function is crucial - it constructs a detailed prompt that explicitly tells the model to use the correct job title and company, reference specific resume bullets, and avoid inventing information. The prompt includes the retrieved chunks organized into 'KEY JOB QUALIFICATIONS' and 'MOST RELEVANT EXPERIENCES' sections, guiding the model to connect them in full paragraphs.

Finally, `generate` orchestrates the full pipeline: retrieval, prompt construction, generation with beam search, and post-processing. It includes a fallback mechanism that uses a rule-based system if the model output is low quality, and filters out sentences that make untrue claims about degrees or experience.

The `inference.py` module handles loading the trained LoRA model and initializing the RAG pipeline, wrapping everything in a clean interface."

---

### [4:30-5:00] Evaluation & Ablation Study
**Files to show:**
- `scripts/evaluate_models.py` (show evaluate_system function, lines 114-175, and main function showing systems, lines 238-302)
- `experiments/results/metrics.json` (show the results)
- `src/inference.py` (show ablation variants: generate_lora_only_model lines 80-109, generate_rag_only_model lines 149-163)

**Script:**
"For evaluation, I implemented a comprehensive metrics suite. The `evaluate_models.py` script evaluates multiple systems: a template baseline that just plugs text into a template, a prompt-only baseline using T5-base without fine-tuning, and our full RAG+LoRA system.

I also implemented ablation variants: LoRA-only without RAG, and RAG-only without LoRA fine-tuning, to understand the contribution of each component.

I measure ROUGE-L for generation quality, and cosine similarity between the generated letter and both the job description and resume to measure relevance and alignment.

As you can see in the results, our RAG+LoRA system achieves a ROUGE-L of 0.27, with job relevance of 0.59 and resume alignment of 0.53. Interestingly, the template baseline scores higher on ROUGE-L, but this is misleading - it's essentially copying structure from the reference. Our system generates more personalized, truthful content. The prompt-only baseline performs very poorly, showing the importance of both fine-tuning and retrieval."

---

### [5:00-5:30] Demo & Web Application
**Files to show:**
- `scripts/run_demo_cli.py` (entire file)
- `app_streamlit.py` (show main structure, lines 1-50, 100-150 if time permits)

**Script:**
"Finally, I've built two interfaces for the system. The CLI demo script provides a simple command-line interface where users can provide resume and job description files and get a generated cover letter.

There's also a Streamlit web application that provides a user-friendly interface with a two-column layout, loading indicators, and download functionality. Both interfaces use the same inference pipeline, ensuring consistency.

The system is production-ready with proper error handling, loading indicators, and fallback mechanisms to ensure users always get a usable output. The first generation may take 30-60 seconds as the model loads, but subsequent generations are faster."

---

### [5:30-6:00] Conclusion & Key Takeaways
**Files to show:**
- Project root (final overview)
- `README.md` (results table again)

**Script:**
"To summarize, this project demonstrates several important ML engineering practices: modular code design, proper train/val/test splits, comprehensive evaluation with ablation studies, and production-ready interfaces.

The key innovations are the combination of RAG for grounding generation in relevant qualifications and experiences, LoRA for efficient fine-tuning, and careful prompt engineering to ensure truthful, personalized output. The system outperforms naive baselines and provides a practical solution for cover letter generation.

The retrieval strategy is particularly interesting - rather than retrieving from a corpus of examples, we retrieve the most relevant parts of the user's own resume and job description, creating a more direct connection between qualifications and experiences.

Thank you for watching! You can find the full code, documentation, and results in the repository."

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
11. `scripts/run_demo_cli.py`
12. `app_streamlit.py` (optional)

**Tips for recording:**
- Use a code editor with good syntax highlighting (VS Code recommended)
- Zoom in to 120-150% for readability
- Use smooth scrolling between files
- Pause briefly when switching files to let viewers orient
- Highlight specific lines when discussing them (use cursor or selection)
- Show the actual results in `metrics.json` when discussing evaluation

**Timing notes:**
- Total script: ~6 minutes when spoken at normal pace
- Adjust speed based on your speaking pace
- It's okay to go slightly over 6 minutes if needed for clarity
- Focus on the RAG pipeline section (3:30-4:30) as it's the most innovative part

**Key talking points to emphasize:**
1. **"LoRA allows efficient fine-tuning with only 16 rank parameters"**
2. **"Classic RAG retrieval over the user's own resume and job description"**
3. **"Instruction-tuning format for better instruction following"**
4. **"Comprehensive evaluation with ablation studies"**
5. **"Rule-based fallback ensures minimum-quality output"**
