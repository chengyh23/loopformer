"""
Inference utilities for LoopFormer / GPT-2 models.
Mirrors the vLLM-style sample() API; tiktoken has no chat_template so
message dicts are formatted as plain "Role: content\n..." text.
"""

import torch
import tiktoken
from contextlib import nullcontext
from tqdm import tqdm


def _apply_chat_template(messages: list[dict]) -> str:
    """Minimal chat-template substitute for GPT-2 (no special role tokens)."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("Assistant:")
    return "\n".join(parts)


def sample(
    model,
    enc: tiktoken.Encoding,
    conversations: list,
    top_k: int = 200,
    max_new_tokens: int = 500,
    temperature: float = 1.0,
    device: str | None = None,
    dtype: str = "bfloat16",
) -> tuple[list[str], list[str]]:
    """
    Generate completions for a batch of conversations or plain-text prompts.

    Args:
        model: GPT / LoopFormer model instance (will be set to eval mode).
        enc: tiktoken encoding, e.g. tiktoken.get_encoding("gpt2").
        conversations: list of either —
            • message lists: [{"role": "user", "content": "..."}, ...]
            • plain strings: used as-is
        top_k: top-k sampling cutoff (use None to disable).
        max_new_tokens: max tokens to generate per prompt.
        temperature: sampling temperature (1.0 = no change).
        device: torch device string (auto-detected from model if None).
        dtype: "bfloat16" / "float16" / "float32".

    Returns:
        texts:   list of formatted input prompts fed to the model.
        answers: list of generated completions (prompt and EOS stripped).
    """
    if device is None:
        device = next(model.parameters()).device

    eos_id = enc._special_tokens.get("<|endoftext|>")
    stop_token_ids = [eos_id] if eos_id is not None else None

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda ids: enc.decode(ids)

    texts = []
    for conv in conversations:
        if isinstance(conv, str):
            texts.append(conv)
        else:
            texts.append(_apply_chat_template(conv))

    answers = []
    model.eval()
    with torch.no_grad():
        with ctx:
            for text in tqdm(texts, desc="Generating"):
                prompt_ids = encode(text)
                x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]
                y = model.generate(
                    x,
                    max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    stop_token_ids=stop_token_ids,
                )
                completion_ids = y[0, len(prompt_ids):].tolist()
                # strip trailing EOS (skip_special_tokens equivalent)
                if stop_token_ids and completion_ids and completion_ids[-1] in stop_token_ids:
                    completion_ids = completion_ids[:-1]
                answers.append(decode(completion_ids))

    return texts, answers


def sample_hf(
    model,
    tokenizer,
    conversations: list,
    top_k: int = 200,
    max_new_tokens: int = 500,
    temperature: float = 1.0,
    device: str | None = None,
) -> tuple[list[str], list[str]]:
    """
    Generate completions for HuggingFace causal LM models (e.g., Ouro).

    Args:
        model: HuggingFace AutoModelForCausalLM instance.
        tokenizer: HuggingFace tokenizer.
        conversations: list of either —
            • message lists: [{"role": "user", "content": "..."}, ...]
            • plain strings: used as-is
        top_k: top-k sampling cutoff (use None to disable).
        max_new_tokens: max tokens to generate per prompt.
        temperature: sampling temperature (1.0 = no change).
        device: torch device string (auto-detected from model if None).

    Returns:
        texts:   list of formatted input prompts fed to the model.
        answers: list of generated completions (prompt and EOS stripped).
    """
    if device is None:
        device = next(model.parameters()).device

    texts = []
    for conv in conversations:
        if isinstance(conv, str):
            texts.append(conv)
        else:
            texts.append(_apply_chat_template(conv))

    answers = []
    model.eval()
    with torch.no_grad():
        for text in tqdm(texts, desc="Generating"):
            inputs = tokenizer(text, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": True,
            }
            if top_k is not None:
                gen_kwargs["top_k"] = top_k
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask

            output_ids = model.generate(input_ids, **gen_kwargs)
            completion_ids = output_ids[0, input_ids.shape[1]:].tolist()

            # strip trailing EOS token
            if completion_ids and completion_ids[-1] == tokenizer.eos_token_id:
                completion_ids = completion_ids[:-1]

            answer = tokenizer.decode(completion_ids, skip_special_tokens=True)
            answers.append(answer)

    return texts, answers

