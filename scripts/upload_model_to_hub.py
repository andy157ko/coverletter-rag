"""
Upload trained model to HuggingFace Hub for Streamlit Cloud deployment.

Usage:
    python scripts/upload_model_to_hub.py --repo_id your-username/coverletter-rag-lora
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, login
import os

def upload_model(repo_id: str, model_path: str = "experiments/logs/rag_lora/best_model.pt"):
    """Upload the trained LoRA model to HuggingFace Hub."""
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Uploading model from {model_path} to {repo_id}...")
    
    # Login (will prompt if not already logged in)
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        login(token=token)
    else:
        login()  # Will prompt for token
    
    # Upload file
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo="best_model.pt",
        repo_id=repo_id,
        repo_type="model",
    )
    
    print(f"✅ Model uploaded successfully to https://huggingface.co/{repo_id}")
    print(f"\nTo use in Streamlit Cloud, update src/inference.py to load from: {repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., 'your-username/coverletter-rag-lora')"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="experiments/logs/rag_lora/best_model.pt",
        help="Path to model file"
    )
    args = parser.parse_args()
    
    upload_model(args.repo_id, args.model_path)

