"""LoRA SFT of the looped Llama on GSM8K, from meta-llama/Llama-3.2-3B-Instruct.

The looped model shares weights across loops. By default the LoRA adapter
attached to the projection modules is likewise shared across every loop
(parameter-efficient looped fine-tuning); with --per-loop-lora, each loop
instead gets its own adapter, switched by a pre-forward hook on every decoder
layer. Activation memory scales with num_loops, so gradient checkpointing is
enabled (LlamaDecoderLayer is a GradientCheckpointingLayer, so it works with
our custom forward automatically; the per-loop adapter name travels as a layer
kwarg, so the backward-pass replay re-activates the right adapter).

Run (single GPU):
    CUDA_VISIBLE_DEVICES=6 python train/train_lora_llamalooped.py \
        --model meta-llama/Llama-3.2-3B-Instruct [--per-loop-lora]

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

# Sibling import (train/ is on sys.path when this script is run directly;
# `train.common` would be shadowed by the top-level train.py).
from common import parse_args, prc_skip_layers

# ---- config ----
RECUR_MODE = "latent"  # "latent" | "latent_renorm" | "token" | "soft"
SKIP_LAYERS = None
OUTPUT_DIR_BASE = "ckpts"  # actual dir: {OUTPUT_DIR_BASE}/{model_short}_looped_lora
MAX_LEN = 1024

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Sandwich-norm submodule names (see LlamaLoopedDecoderLayer,
# use_loop_sandwichnorm); trained fully (not via LoRA) when --sandwich-norm.
SANDWICH_NORM_NAMES = ("ln_attn_inner", "ln_mlp_inner", "post_attn_ln", "post_mlp_ln")

# AttnRes submodule name (see LoopAttnResidual); trained fully (not via LoRA)
# and saved separately when --loop-attn-res.
LOOP_ATTN_RES_NAME = "loop_attn_res"

# TrainingArguments
EPOCHS = 1
LR = 2e-4

# Weights & Biases (pip install wandb; `wandb login` once).
USE_WANDB = True
WANDB_PROJECT = "llama_looped_lora"


def build_dataset(tok, n_train=None):
    ds = load_dataset("gsm8k", "main", split="train")
    if n_train:
        ds = ds.select(range(n_train))

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
    args_cli = parse_args(__doc__)
    model_name = args_cli.model
    model_short = model_name.rstrip("/").split("/")[-1]
    per_loop_lora = args_cli.per_loop_lora
    num_loops = args_cli.num_loops
    prc = args_cli.prc  # None or [prelude, coda]
    suffix = (
        (f"_L{num_loops}" if num_loops != 4 else "")
        + ("_perloop" if per_loop_lora else "")
        + (f"_prc{prc[0]}-{prc[1]}" if prc else "")
        + ("_sandwich" if args_cli.sandwich_norm else "")
    )
    output_dir = os.path.join(OUTPUT_DIR_BASE, f"{model_short}_looped_lora{suffix}")
    run_name = (
        f"{model_short}-gsm8k-lora-{RECUR_MODE}-L{num_loops}-r{LORA_R}"
        + suffix.replace("_", "-")
    )

    if USE_WANDB:
        os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # right-pad for causal-LM training

    train_ds = build_dataset(tok, args_cli.n_train)

    model = LlamaLoopedForCausalLM.from_pretrained(
        model_name,
        num_loops=num_loops,
        recur_mode=RECUR_MODE,
        skip_layers=SKIP_LAYERS,
        use_loop_sandwichnorm=args_cli.sandwich_norm,
        use_loop_attn_residual=args_cli.loop_attn_res,
        dtype=torch.bfloat16,
    )
    if prc:
        skip = prc_skip_layers(
            prc[0], prc[1], model.config.num_hidden_layers, num_loops
        )
        model.set_skip_layers(skip)
        print(f"[INFO] P-R-C skip_layers: {skip}")
    model.config.use_cache = False  # required with gradient checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # needed for grad-ckpt + LoRA

    def make_lora():
        return LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGETS,
            bias="none",
            task_type="CAUSAL_LM",
        )

    if per_loop_lora:
        adapter_names = [f"loop{i}" for i in range(num_loops)]
        model = get_peft_model(model, make_lora(), adapter_name=adapter_names[0])
        for name in adapter_names[1:]:
            model.add_adapter(name, make_lora())
        # Every adapter must have requires_grad=True when the Trainer builds
        # the optimizer; during forward, the per-layer pre-hook activates the
        # adapter belonging to the current loop.
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)
        model.get_base_model().set_loop_adapters(adapter_names)
    else:
        adapter_names = None
        model = get_peft_model(model, make_lora())
    if args_cli.sandwich_norm:
        # Sandwich norms are newly initialized (not in the pretrained
        # checkpoint) and shared across loops (same nn.Module reused every
        # loop), so they're trained fully here rather than via LoRA/
        # modules_to_save; get_peft_model froze them along with the rest of
        # the base model, so unfreeze them explicitly.
        for n, p in model.named_parameters():
            if any(k in n for k in SANDWICH_NORM_NAMES):
                p.requires_grad_(True)
    if args_cli.loop_attn_res:
        # AttnRes weights are newly initialized (not in the pretrained
        # checkpoint) and were frozen by get_peft_model along with the base
        # model; train them fully, like the sandwich norms above.
        for n, p in model.named_parameters():
            if LOOP_ATTN_RES_NAME in n:
                p.requires_grad_(True)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args_cli.batch_size,
        gradient_accumulation_steps=args_cli.grad_accum,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=(["wandb"] if USE_WANDB else "none"),
        run_name=run_name,
        # Batch similar lengths together (less padding); lengths are computed
        # from input_ids since this dataset has no "length" column.
        train_sampling_strategy="group_by_length",
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

    adapter_dir = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    print(f"saved LoRA adapter to: {adapter_dir}")
    if args_cli.sandwich_norm:
        # model.save_pretrained (peft) only persists LoRA params, not the
        # manually-unfrozen sandwich norms, so save those separately.
        norm_sd = {
            k: v
            for k, v in model.state_dict().items()
            if any(s in k for s in SANDWICH_NORM_NAMES)
        }
        norm_path = os.path.join(adapter_dir, "sandwich_norms.pt")
        torch.save(norm_sd, norm_path)
        print(f"saved sandwich-norm weights to: {norm_path}")
    if args_cli.loop_attn_res:
        # peft's save_pretrained only persists LoRA params; save the fully-
        # trained AttnRes weights separately.
        attn_res_sd = {
            k: v
            for k, v in model.state_dict().items()
            if LOOP_ATTN_RES_NAME in k
        }
        attn_res_path = os.path.join(adapter_dir, "loop_attn_res.pt")
        torch.save(attn_res_sd, attn_res_path)
        print(f"saved loop-attn-res weights to: {attn_res_path}")
    load_hint = (
        "load with:\n"
        f"  base = LlamaLoopedForCausalLM.from_pretrained('{model_name}',"
        f" num_loops={num_loops}, recur_mode='{RECUR_MODE}',"
        f" use_loop_sandwichnorm={args_cli.sandwich_norm},"
        f" use_loop_attn_residual={args_cli.loop_attn_res}, dtype='bfloat16')\n"
        "  from peft import PeftModel\n"
    )
    if per_loop_lora:
        # save_pretrained wrote one subdir per adapter (loop0/, loop1/, ...).
        load_hint += (
            f"  model = PeftModel.from_pretrained(base,"
            f" '{adapter_dir}/{adapter_names[0]}', adapter_name='{adapter_names[0]}')\n"
        )
        for name in adapter_names[1:]:
            load_hint += (
                f"  model.load_adapter('{adapter_dir}/{name}', adapter_name='{name}')\n"
            )
        load_hint += f"  model.get_base_model().set_loop_adapters({adapter_names!r})"
    else:
        load_hint += f"  model = PeftModel.from_pretrained(base, '{adapter_dir}')"
    if args_cli.sandwich_norm:
        load_hint += (
            f"\n  model.load_state_dict(torch.load('{norm_path}'), strict=False)"
        )
    if args_cli.loop_attn_res:
        load_hint += (
            f"\n  model.load_state_dict(torch.load('{attn_res_path}'), strict=False)"
        )
    print(load_hint)


if __name__ == "__main__":
    main()
