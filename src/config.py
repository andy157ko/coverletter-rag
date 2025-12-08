from dataclasses import dataclass

@dataclass
class TrainingConfig:
    model_name: str = "google/flan-t5-base"
    max_input_length: int = 1024
    max_target_length: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    batch_size: int = 4
    num_epochs: int = 3
    max_input_length: int = 1024
    max_target_length: int = 512
    warmup_steps: int = 100
    output_dir: str = "experiments/logs"

@dataclass
class DataConfig:
    train_path: str = "data/processed/train.jsonl"
    val_path: str = "data/processed/val.jsonl"
    test_path: str = "data/processed/test.jsonl"

@dataclass
class RAGConfig:
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_path: str = "data/embeddings/faiss_index.bin"
    k_retrieval: int = 5

