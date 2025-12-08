# Quick Navigation Reference for Video Recording

## File Click Order (Follow this sequence)

### 1. Introduction [0:00]
**Click:** `README.md`
- Show lines 1-20
- Talk about project overview

### 2. Architecture [0:30]
**Click:** `src/config.py`
- Show all three dataclass definitions
- Explain TrainingConfig, DataConfig, RAGConfig

### 3. Data Pipeline [1:30]
**Click:** `scripts/build_original_dataset.py`
- Scroll to main() function (lines 33-68)
- Show how data is processed

**Click:** `data/processed/train.jsonl`
- Show 1-2 example lines
- Explain JSONL format

**Click:** `scripts/build_embeddings.py`
- Show entire file (it's short)
- Explain FAISS index creation

### 4. Model & Training [2:30]
**Click:** `src/models.py`
- Show entire file
- Explain LoRA application

**Click:** `src/train.py`
- Show lines 29-87 (collate function)
- Show lines 90-140 (training setup)
- Show lines 142-180 (training loop)
- Explain regularization, early stopping

### 5. RAG Pipeline [3:30]
**Click:** `src/rag_pipeline.py`
- Show lines 20-53 (extract_title_and_company)
- Show lines 56-117 (extract_resume_bullets)
- Show lines 264-296 (retrieve method)
- Show lines 298-328 (build_prompt)
- Show lines 331-410 (generate method)
- Explain retrieval, prompt engineering, fallback

**Click:** `src/inference.py`
- Show lines 67-128 (_init_rag_pipeline)
- Explain model loading

### 6. Evaluation [4:30]
**Click:** `scripts/evaluate_models.py`
- Show lines 66-125 (evaluate_system function)
- Explain metrics

**Click:** `experiments/results/metrics.json`
- Show the results table
- Compare baselines vs RAG+LoRA

### 7. Demo [5:00]
**Click:** `app_streamlit.py`
- Show lines 1-50 (imports and setup)
- Show lines 100-150 (UI components)
- Explain web interface

**Click:** `scripts/run_demo_cli.py`
- Show entire file
- Explain CLI option

### 8. Conclusion [5:30]
**Click:** Project root (file explorer)
- Show overall structure
- Wrap up

---

## Pro Tips

✅ **Before recording:**
- Open all files in separate tabs
- Test your screen recording software
- Adjust font size to 14-16pt for readability
- Set editor theme to light/dark based on preference

✅ **During recording:**
- Use smooth scrolling (not jumpy)
- Pause 1-2 seconds when switching files
- Use cursor to highlight specific lines
- Speak clearly and at moderate pace

✅ **If you go over time:**
- Cut the demo section shorter if needed
- Focus on core ML components (training, RAG pipeline)
- Evaluation can be summarized quickly

---

## Key Talking Points (Memorize These)

1. **"LoRA allows efficient fine-tuning with only 16 rank parameters"**
2. **"FAISS enables fast semantic retrieval of top-k similar examples"**
3. **"Prompt engineering ensures truthful, personalized output"**
4. **"Early stopping and regularization prevent overfitting"**
5. **"RAG+LoRA outperforms baselines: 0.324 ROUGE-L vs 0.215"**

