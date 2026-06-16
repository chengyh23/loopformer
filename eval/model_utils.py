"""
Shared model loading utilities for evaluation scripts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tiktoken
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_loopformer(model_path: str, device: str = "cuda"):
    """
    Load a LoopFormer model and return model + tokenizer.

    Args:
        model_path: HuggingFace model ID or local checkpoint path
        device: torch device to place model on

    Returns:
        (model, tokenizer): GPT model and tiktoken encoder
    """
    from models.loopformer import GPT

    model = GPT.from_pretrained(model_path)
    model.to(device)
    tokenizer = tiktoken.get_encoding("gpt2")
    return model, tokenizer


def load_model_ouro(model_path: str = "ByteDance/Ouro-1.4B", device: str = "cuda"):
    """
    Load an Ouro (or any HuggingFace causal LM) model.

    Args:
        model_path: HuggingFace model ID (default: ByteDance/Ouro-1.4B)
        device: torch device

    Returns:
        (model, tokenizer): HuggingFace model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        dtype="auto",
        trust_remote_code=True,
    )
    return model, tokenizer
