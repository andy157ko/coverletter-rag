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


_RAG_EXPERIMENT_NAME = "rag_lora"

_config = TrainingConfig()
_LOG_DIR = Path(_config.output_dir) / _RAG_EXPERIMENT_NAME
_BEST_MODEL_PATH = _LOG_DIR / "best_model.pt"

# Option to load model from HuggingFace Hub (for Streamlit Cloud)
# Set this to your Hub repo ID if you've uploaded the model there
# Example: "your-username/coverletter-rag-lora"
_MODEL_HUB_REPO = os.getenv("MODEL_HUB_REPO", None)

_INDEX_PATH = Path("data/embeddings/faiss_index.bin")
_METADATA_PATH = Path("data/embeddings/metadata.jsonl")

_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)



_rag_pipeline = None  


def _load_metadata(metadata_path: Path):
    records = []
    with metadata_path.open() as f:
        for line in f:
            records.append(json.loads(line))
    return records

def _init_lora_only_model():
    """
    Load ONLY the LoRA fine-tuned model, without
    building the RAG retrieval index.
    """
    tokenizer_name = _config.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_auth_token=_HF_TOKEN if _HF_TOKEN else None
    )

    base_model = load_base_model(
        tokenizer_name,
        is_seq2seq=True,
    )

    lora_model = apply_lora(
        base_model,
        r=_config.lora_r,
        alpha=_config.lora_alpha,
        dropout=_config.lora_dropout,
    )

    if not _BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {_BEST_MODEL_PATH}. Run training first."
        )

    state_dict = torch.load(_BEST_MODEL_PATH, map_location="cpu")
    lora_model.load_state_dict(state_dict)
    lora_model.eval()

    return tokenizer, lora_model


_lora_only_cache = None


def generate_lora_only_model(resume_text: str, job_text: str) -> str:
    """
    Use ONLY the LoRA fine-tuned seq2seq model.
    No retrieval, no RAG context.
    """
    global _lora_only_cache

    if _lora_only_cache is None:
        _lora_only_cache = _init_lora_only_model()

    tokenizer, model = _lora_only_cache

    prompt = (
        "You are an expert career coach helping students apply to jobs.\n"
        "Write a personalized cover letter.\n\n"
        f"TARGET JOB:\n{job_text}\n\n"
        f"TARGET RESUME:\n{resume_text}\n\n"
        "Cover letter:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=_config.max_target_length,
            num_beams=4,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def _init_rag_only_pipeline():
    """
    Build a RAG pipeline using the base pretrained model WITHOUT LoRA weights.
    """
    tokenizer_name = _config.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_auth_token=_HF_TOKEN if _HF_TOKEN else None,
    )

    base_model = load_base_model(
        tokenizer_name,
        is_seq2seq=True,
    )
    base_model.eval()

    # Load FAISS + metadata
    if not _INDEX_PATH.exists():
        raise FileNotFoundError("FAISS index missing. Run build_embeddings.py.")

    index = faiss.read_index(str(_INDEX_PATH))
    metadata_store = _load_metadata(_METADATA_PATH)

    # Build RAG pipeline WITH base model (no LoRA)
    rag_only_pipe = RAGPipeline(
        generator=base_model,
        tokenizer_name=tokenizer_name,
        embed_model_name=_EMBED_MODEL_NAME,
        index=index,
        metadata_store=metadata_store,
    )

    return rag_only_pipe


_rag_only_cache = None


def generate_rag_only_model(resume_text: str, job_text: str) -> str:
    """
    Use RAG retrieval but NO LoRA fine-tuning.
    """
    global _rag_only_cache

    if _rag_only_cache is None:
        _rag_only_cache = _init_rag_only_pipeline()

    return _rag_only_cache.generate(
        resume_text,
        job_text,
        k=5,  # same retrieval count for fair comparison
        max_new_tokens=_config.max_target_length,
    )


def _init_rag_pipeline():
    """
    Initialize the RAG pipeline once and cache it in a module-level variable.
    """
    global _rag_pipeline

    if _rag_pipeline is not None:
        return _rag_pipeline

    #Loading tokenizer and base model
    tokenizer_name = _config.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_auth_token=_HF_TOKEN if _HF_TOKEN else None,
    )

    base_model = load_base_model(
        tokenizer_name,
        is_seq2seq=True,  # our fine-tuned model is seq2seq (T5-style)
    )

    # Applying LoRA configuration (same as in training)
    lora_model = apply_lora(
        base_model,
        r=_config.lora_r,
        alpha=_config.lora_alpha,
        dropout=_config.lora_dropout,
    )

    #Loading fine-tuned weights
    # Try to load from HuggingFace Hub first (for Streamlit Cloud), then local file
    if _MODEL_HUB_REPO:
        try:
            from huggingface_hub import hf_hub_download
            print(f"Loading model from HuggingFace Hub: {_MODEL_HUB_REPO}")
            model_file = hf_hub_download(
                repo_id=_MODEL_HUB_REPO,
                filename="best_model.pt",
                token=_HF_TOKEN,
            )
            state_dict = torch.load(model_file, map_location="cpu")
            print("✅ Model loaded from HuggingFace Hub")
        except Exception as e:
            print(f"Failed to load from Hub: {e}")
            print("Falling back to local file...")
            if not _BEST_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Could not find model locally ({_BEST_MODEL_PATH}) or on Hub ({_MODEL_HUB_REPO}). "
                    "Make sure you've uploaded the model to HuggingFace Hub or trained it locally."
                )
            state_dict = torch.load(_BEST_MODEL_PATH, map_location="cpu")
    else:
        if not _BEST_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Could not find {_BEST_MODEL_PATH}. "
                "Make sure you ran src/train.py with config_name='rag_lora', "
                "or set MODEL_HUB_REPO environment variable to load from HuggingFace Hub."
            )
        state_dict = torch.load(_BEST_MODEL_PATH, map_location="cpu")
    
    lora_model.load_state_dict(state_dict)
    lora_model.eval()  # inference mode

    #sanity-check LoRA presence
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

    #Building RAG pipeline
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
    # k = number of retrieved examples
    return pipeline.generate(
        resume_text,
        job_text,
        k=5,
        max_new_tokens=_config.max_target_length,
    )
