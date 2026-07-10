"""LoRA SFT of the looped Llama on GSM8K, from meta-llama/Llama-3.2-3B-Instruct.

The looped model shares weights across loops, so the LoRA adapter attached to the
projection modules is likewise shared across every loop (parameter-efficient
looped fine-tuning). Activation memory scales with num_loops, so gradient
checkpointing is enabled (LlamaDecoderLayer is a GradientCheckpointingLayer, so
it works with our custom forward automatically).

Run (single GPU):
    CUDA_VISIBLE_DEVICES=6 python train/train_lora_llamalooped.py

Loss is logged to Weights & Biases when USE_WANDB=True (pip install wandb).
"""

import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# Make `models` importable when run as `python tests/train_lora_llamalooped.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llama_looped import LlamaLoopedForCausalLM

# ---- config ----
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
NUM_LOOPS = 4
RECUR_MODE = "latent"  # "latent" | "latent_renorm" | "token" | "soft"
SKIP_LAYERS = None
OUTPUT_DIR = "ckpts/llama_looped_lora"
MAX_LEN = 1024
N_TRAIN = None  # e.g. 2000 for a quick run; None = full train split

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# TrainingArguments
EPOCHS = 1
LR = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM = 16

# Weights & Biases (pip install wandb; `wandb login` once).
USE_WANDB = True
WANDB_PROJECT = "llama_looped_lora"
RUN_NAME = f"gsm8k-lora-{RECUR_MODE}-L{NUM_LOOPS}-r{LORA_R}"


def build_dataset(tok):
    ds = load_dataset("gsm8k", "main", split="train")
    if N_TRAIN:
        ds = ds.select(range(N_TRAIN))

    def format_example(ex):
        # Prompt = chat-templated user turn (masked in the loss); response =
        # the full GSM8K answer (reasoning + "#### <final>") + EOS.
        prompt_ids = tok.apply_chat_template(
            [{"role": "user", "content": ex["question"]}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        resp_ids = tok(ex["answer"], add_special_tokens=False)["input_ids"]
        resp_ids = resp_ids + [tok.eos_token_id]
        input_ids = (prompt_ids + resp_ids)[:MAX_LEN]
        labels = ([-100] * len(prompt_ids) + resp_ids)[:MAX_LEN]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

    return ds.map(format_example, remove_columns=ds.column_names)


def main():
    if USE_WANDB:
        os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # right-pad for causal-LM training

    train_ds = build_dataset(tok)

    model = LlamaLoopedForCausalLM.from_pretrained(
        MODEL,
        num_loops=NUM_LOOPS,
        recur_mode=RECUR_MODE,
        skip_layers=SKIP_LAYERS,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False  # required with gradient checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # needed for grad-ckpt + LoRA

    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=(["wandb"] if USE_WANDB else "none"),
        run_name=RUN_NAME,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(
            tok, padding=True, label_pad_token_id=-100
        ),
    )
    trainer.train()

    adapter_dir = os.path.join(OUTPUT_DIR, "adapter")
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    print(f"saved LoRA adapter to: {adapter_dir}")
    print(
        "load with:\n"
        "  base = LlamaLoopedForCausalLM.from_pretrained(MODEL, num_loops=NUM_LOOPS,"
        " recur_mode=RECUR_MODE, dtype='bfloat16')\n"
        "  from peft import PeftModel; model = PeftModel.from_pretrained(base,"
        f" '{adapter_dir}')"
    )


if __name__ == "__main__":
    main()
