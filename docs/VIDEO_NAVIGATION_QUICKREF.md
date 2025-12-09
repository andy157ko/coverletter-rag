# Quick Navigation Reference for Video Recording

## File Click Order (Follow this sequence)

### 1. Introduction [0:00]
**Click:** `README.md`
- Show lines 1-20 (project description)
- Show lines 76-98 (results table)
- Talk about project overview and results

### 2. Architecture [0:30]
**Click:** `src/config.py`
- Show entire file (all three dataclasses)
- Explain TrainingConfig, DataConfig, RAGConfig
- Highlight LoRA parameters and model choice

### 3. Data Pipeline [1:30]
**Click:** `scripts/build_original_dataset.py`
- Show build_instruction_example function (lines 34-62)
- Show main() function (lines 76-112)
- Explain instruction-tuning format conversion

**Click:** `data/processed/train.jsonl`
- Show 1-2 example lines
- Point out "input" and "output" fields
- Explain JSONL format

**Click:** `scripts/build_embeddings.py`
- Show entire file (it's short)
- Explain FAISS index creation
- Note: embeddings are for training corpus, not used in final RAG

### 4. Model & Training [2:30]
**Click:** `src/models.py`
- Show entire file
- Explain LoRA application
- Highlight parameter efficiency

**Click:** `src/train.py`
- Show lines 29-67 (collate function - instruction format handling)
- Show lines 71-97 (training setup - optimizer, scheduler)
- Show lines 123-180 (training loop - regularization, early stopping)
- Explain gradient clipping, weight decay, TensorBoard logging

### 5. RAG Pipeline [3:30] - **MOST IMPORTANT SECTION**
**Click:** `src/rag_pipeline.py`
- Show lines 17-42 (extract_title_and_company)
- Show lines 45-92 (extract_resume_bullets)
- Show lines 220-296 (retrieve method - **KEY INNOVATION**)
  - Explain: retrieval over user's own resume/job, not training corpus
  - Show chunking strategy
  - Show cosine similarity computation
- Show lines 298-390 (build_prompt - **CRITICAL**)
  - Show how retrieved chunks are organized
  - Show explicit rules about truthfulness
  - Show job/resume bullet integration
- Show lines 393-510 (generate method)
  - Show beam search parameters
  - Show post-processing and filtering
  - Show fallback mechanism

**Click:** `src/inference.py`
- Show lines 166-232 (_init_rag_pipeline)
- Explain model loading and pipeline initialization
- Show LoRA parameter verification

### 6. Evaluation [4:30]
**Click:** `scripts/evaluate_models.py`
- Show lines 114-175 (evaluate_system function)
- Show lines 238-302 (main function - systems definition)
- Explain metrics: ROUGE-L, job relevance, resume alignment
- Show ablation variants

**Click:** `src/inference.py`
- Show lines 80-109 (generate_lora_only_model)
- Show lines 149-163 (generate_rag_only_model)
- Explain ablation study components

**Click:** `experiments/results/metrics.json`
- Show the results table
- Compare all systems
- Explain why template baseline ROUGE-L is misleading

### 7. Demo [5:00]
**Click:** `scripts/run_demo_cli.py`
- Show entire file
- Explain CLI interface

**Click:** `app_streamlit.py` (if time permits)
- Show lines 1-50 (imports and setup)
- Show lines 100-150 (UI components)
- Explain web interface

### 8. Conclusion [5:30]
**Click:** Project root (file explorer)
- Show overall structure
- Wrap up with key takeaways

---

## Pro Tips

✅ **Before recording:**
- Open all files in separate tabs in order
- Test your screen recording software
- Adjust font size to 14-16pt for readability
- Set editor theme to light/dark based on preference
- Have `metrics.json` open to show actual results

✅ **During recording:**
- Use smooth scrolling (not jumpy)
- Pause 1-2 seconds when switching files
- Use cursor to highlight specific lines
- Speak clearly and at moderate pace
- **Emphasize the RAG retrieval innovation** (retrieval over user's own text, not corpus)

✅ **If you go over time:**
- Cut the demo section shorter if needed
- Focus heavily on RAG pipeline section (it's the most innovative)
- Evaluation can be summarized quickly
- Skip Streamlit app if needed

---

## Key Talking Points (Memorize These)

1. **"LoRA allows efficient fine-tuning with only 16 rank parameters per layer"**
2. **"Classic RAG retrieval over the user's own resume and job description, not a training corpus"**
3. **"Instruction-tuning format ensures the model follows detailed instructions"**
4. **"Comprehensive evaluation with ablation studies shows contribution of each component"**
5. **"Rule-based fallback ensures minimum-quality output even when model struggles"**
6. **"ROUGE-L of 0.27 with strong job relevance (0.59) and resume alignment (0.53)"**

---

## Critical Sections to Emphasize

1. **RAG Retrieval Strategy** (lines 220-296 in rag_pipeline.py)
   - This is the key innovation
   - Explain why retrieving from user's own text is better than corpus retrieval
   - Show the chunking and similarity computation

2. **Prompt Engineering** (lines 298-390 in rag_pipeline.py)
   - Show how retrieved chunks are organized
   - Show explicit truthfulness rules
   - Explain how this prevents hallucination

3. **Ablation Study** (inference.py + evaluate_models.py)
   - Shows understanding of component contributions
   - Demonstrates scientific rigor
