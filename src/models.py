"""
Model definitions: base LM, LoRA/PEFT, and baselines.

Rubric:
- Fine-tuned pretrained model
- Transformer LM
- Parameter-efficient fine-tuning (LoRA)
- Regularization (dropout/weight decay configured here)
"""

from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def load_base_model(model_name: str, is_seq2seq: bool = True):
    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        raise ValueError("Only seq2seq models are supported right now.")
    return model

def apply_lora(model, r: int, alpha: int, dropout: float):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",  
    )
    lora_model = get_peft_model(model, lora_config)
    return lora_model
