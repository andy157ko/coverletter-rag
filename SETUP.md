# SETUP

## 1. Environment Setup

### Option A: Conda (Recommended - Fixes Python 3.13 compatibility issues)

If you're using Anaconda/Miniconda and encountering library conflicts (especially with Python 3.13), use this:

```bash
# Create conda environment with Python 3.11 (more stable with PyTorch)
conda create -n coverletter-rag python=3.11 -y

# Activate the environment
conda activate coverletter-rag

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

Or use the setup script:
```bash
./setup_env.sh
```

### Option B: Virtual Environment (venv)

If you prefer using venv:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** Python 3.11 or 3.12 is recommended. Python 3.13 may have compatibility issues with PyTorch.

## 2. Prepare Data

```bash
# Download and process the dataset
python scripts/build_original_dataset.py

# Build FAISS embeddings index
python scripts/build_embeddings.py
```

## 3. Train the Model

```bash
# Train the LoRA fine-tuned model
python src/train.py --config_name rag_lora
```

This will:
- Load the base model (Flan-T5-base)
- Apply LoRA fine-tuning
- Save the best model to `experiments/logs/rag_lora/best_model.pt`

## 4. Run the Web App

```bash
# Make sure you're in the activated environment
conda activate coverletter-rag  # or: source .venv/bin/activate

# Run Streamlit app
streamlit run app_streamlit.py
```

The app will open at `http://localhost:8501`

## Troubleshooting

### ImportError with PyTorch (Symbol not found)
This usually happens with Python 3.13 or conda base environment conflicts. Solution:
1. Create a new conda environment with Python 3.11 (see Option A above)
2. Install all packages fresh in that environment

### Model Not Found
Make sure you've completed steps 2 and 3 above before running the web app.

### Port Already in Use
```bash
streamlit run app_streamlit.py --server.port 8502
```
