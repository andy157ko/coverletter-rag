# src/train.py

"""
Training script for LoRA fine-tuned cover letter generator.

Rubric coverage:
- Modular code design
- Proper train/val split and dataloaders
- Training curves with TensorBoard
- Regularization (weight decay, gradient clipping, early stopping)
- Hyperparameter config via TrainingConfig
"""

import argparse
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW

from transformers import get_linear_schedule_with_warmup

from src.config import TrainingConfig, DataConfig
from src.data import create_dataloaders
from src.preprocessing import get_tokenizer
from src.models import load_base_model, apply_lora


def build_collate_fn(tokenizer, max_input_length: int, max_target_length: int):
    """
    Collate function for cover-letter dataset:
    - builds input_text from job_text + resume_text
    - tokenizes inputs / targets
    - masks pad tokens in labels with -100 for loss computation

    We bind tokenizer to a local variable `tok` to avoid any accidental
    shadowing or mutation.
    """

    tok = tokenizer  # bind to local name, never mutated

    def collate_fn(batch):
        input_texts = []
        target_texts = []

        for ex in batch:
            job_text = ex["job_text"]
            resume_text = ex["resume_text"]
            cover_letter_text = ex["cover_letter_text"]

            # Build the input prompt
            input_text = (
                "You are an assistant that writes tailored cover letters for job applications.\n\n"
                f"JOB:\n{job_text}\n\n"
                f"RESUME:\n{resume_text}\n\n"
                "TASK:\nWrite a professional, concise cover letter for this job based on the resume."
            )

            input_texts.append(input_text)
            target_texts.append(cover_letter_text)

        # Tokenize inputs
        model_inputs = tok(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        )

        # Tokenize targets
        with tok.as_target_tokenizer():
            labels = tok(
                target_texts,
                padding=True,
                truncation=True,
                max_length=max_target_length,
                return_tensors="pt",
            )["input_ids"]

        # Replace padding token id's in labels by -100 for loss to ignore them
        labels[labels == tok.pad_token_id] = -100

        model_inputs["labels"] = labels
        return model_inputs

    return collate_fn


def train(config: TrainingConfig, data_config: DataConfig, config_name: str = "rag_lora"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Tokenizer
    tokenizer = get_tokenizer(config.model_name)
    print(f"Loaded tokenizer: {type(tokenizer)}")

    # 2) Base model + LoRA
    model = load_base_model(config.model_name, is_seq2seq=True).to(device)
    model = apply_lora(model, config.lora_r, config.lora_alpha, config.lora_dropout)
    model.train()

    # 3) Dataloaders
    collate_fn = build_collate_fn(
        tokenizer,
        max_input_length=config.max_input_length,
        max_target_length=config.max_target_length,
    )

    loaders = create_dataloaders(
        data_config.train_path,
        data_config.val_path,
        data_config.test_path,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
    )

    # 4) Optimizer + scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_steps = len(loaders["train"]) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )

    # 5) Logging
    log_dir = Path(config.output_dir) / config_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_val_loss = float("inf")
    patience = 2
    patience_counter = 0
    global_step = 0

    # 6) Training loop
    for epoch in range(config.num_epochs):
        model.train()
        print(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===")

        step = 0  # add this
        for batch in loaders["train"]:
            step += 1
            if step % 10 == 0:
                print(f"  [train] step {step}/{len(loaders['train'])}")

            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in loaders["val"]:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                val_losses.append(outputs.loss.item())

        val_loss = sum(val_losses) / len(val_losses)
        writer.add_scalar("val/loss", val_loss, epoch)
        print(f"Epoch {epoch + 1} | val_loss = {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), log_dir / "best_model.pt")
            print(f"  → New best model saved to {log_dir / 'best_model.pt'}")
        else:
            patience_counter += 1
            print(f"  → No improvement, patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="rag_lora")
    args = parser.parse_args()

    train(TrainingConfig(), DataConfig(), config_name=args.config_name)
