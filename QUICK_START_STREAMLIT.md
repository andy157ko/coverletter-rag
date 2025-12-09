# Quick Start: Run Streamlit Locally

## Step 1: Activate the Conda Environment

```bash
conda activate coverletter-rag
```

You should see `(coverletter-rag)` in your terminal prompt.

## Step 2: Navigate to Project Directory

```bash
cd /Users/andy/Desktop/coverletter-rag
```

## Step 3: Run Streamlit

**Option A: Using the script (easiest)**
```bash
./run_streamlit.sh
```

**Option B: Direct command**
```bash
streamlit run app_streamlit.py
```

## Step 4: Open in Browser

The app will automatically open in your browser at:
- **http://localhost:8501**

If it doesn't open automatically, copy the URL from the terminal output.

---

## Troubleshooting

### "Command not found: streamlit"
Make sure you've activated the conda environment:
```bash
conda activate coverletter-rag
pip install streamlit  # if not installed
```

### "Model files not found"
Make sure you've:
1. Run `python scripts/build_original_dataset.py`
2. Run `python scripts/build_embeddings.py`
3. Run `python src/train.py --config_name rag_lora`

The model should be at: `experiments/logs/rag_lora/best_model.pt`

### Port 8501 already in use
Use a different port:
```bash
streamlit run app_streamlit.py --server.port 8502
```

### First generation is slow
This is normal! The first generation loads the model into memory (~30-60 seconds). Subsequent generations are faster.

---

## Quick Test

Once the app is running:
1. Paste a sample resume in the left text area
2. Paste a job description in the right text area
3. Click "Generate Cover Letter"
4. Wait for generation (first time will be slower)
5. Review and download the result!

