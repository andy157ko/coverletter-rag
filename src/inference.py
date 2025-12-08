# src/inference.py

from pathlib import Path
import json
import os

import torch
import faiss
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer  # just to ensure dependency present

from src.config import TrainingConfig
from src.models import load_base_model, apply_lora
from src.rag_pipeline import RAGPipeline


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

_RAG_EXPERIMENT_NAME = "rag_lora"

_config = TrainingConfig()
_LOG_DIR = Path(_config.output_dir) / _RAG_EXPERIMENT_NAME
_BEST_MODEL_PATH = _LOG_DIR / "best_model.pt"

_INDEX_PATH = Path("data/embeddings/faiss_index.bin")
_METADATA_PATH = Path("data/embeddings/metadata.jsonl")

_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)


# -----------------------------------------------------------------------------
# Lazy global initialization
# -----------------------------------------------------------------------------

_rag_pipeline = None  # will be initialized on first call


def _load_metadata(metadata_path: Path):
    records = []
    with metadata_path.open() as f:
        for line in f:
            records.append(json.loads(line))
    return records


def _init_rag_pipeline():
    """
    Initialize the RAG pipeline once and cache it in a module-level variable.
    """
    global _rag_pipeline

    if _rag_pipeline is not None:
        return _rag_pipeline

    # 1) Load tokenizer and base model
    tokenizer_name = _config.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_auth_token=_HF_TOKEN if _HF_TOKEN else None,
    )

    base_model = load_base_model(
        tokenizer_name,
        is_seq2seq=True,  # our fine-tuned model is seq2seq (T5-style)
    )

    # 2) Apply LoRA configuration (same as in training)
    lora_model = apply_lora(
        base_model,
        r=_config.lora_r,
        alpha=_config.lora_alpha,
        dropout=_config.lora_dropout,
    )

    # 3) Load fine-tuned weights
    if not _BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {_BEST_MODEL_PATH}. "
            "Make sure you ran src/train.py with config_name='rag_lora'."
        )

    state_dict = torch.load(_BEST_MODEL_PATH, map_location="cpu")
    lora_model.load_state_dict(state_dict)
    lora_model.eval()  # inference mode

    # ---- NEW: sanity-check LoRA presence ----
    lora_param_names = [n for n, _ in lora_model.named_parameters() if "lora" in n.lower()]
    print(f"[DEBUG] LoRA model loaded from {_BEST_MODEL_PATH}")
    print(f"[DEBUG] Found {len(lora_param_names)} LoRA parameter tensors.")
    if lora_param_names:
        print("[DEBUG] Sample LoRA parameter names:", lora_param_names[:10])

    # 4) Load FAISS index and metadata
    if not _INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {_INDEX_PATH}. "
            "Make sure you ran scripts/build_embeddings.py first."
        )

    index = faiss.read_index(str(_INDEX_PATH))
    metadata_store = _load_metadata(_METADATA_PATH)

    # 5) Build RAG pipeline
    _rag_pipeline = RAGPipeline(
        generator=lora_model,
        tokenizer_name=tokenizer_name,
        embed_model_name=_EMBED_MODEL_NAME,
        index=index,
        metadata_store=metadata_store,
    )

    return _rag_pipeline


def generate_rag_lora_model(resume_text: str, job_text: str) -> str:
    """
    Generate a cover letter using the full RAG + LoRA system.

    This is the main entry point used by scripts/evaluate_models.py.
    """
    pipeline = _init_rag_pipeline()
    # k = number of retrieved examples; adjust if you like
    return pipeline.generate(
        resume_text,
        job_text,
        k=5,
        max_new_tokens=_config.max_target_length,
    )
