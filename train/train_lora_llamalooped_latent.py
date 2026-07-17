"""LoRA SFT of the looped Llama on GSM8K-Aug with latent carry + per-loop
generative readout branches ("LoopCoT").

The reasoning steps are removed from the token stream. The input is the
question only; loop i runs the decoder stack over ``[latent(Q); step_i text]``
(teacher forced) and is supervised with CE on the step-i tokens, but ONLY the
Q positions are carried to loop i+1 — the step text is a per-loop readout
branch, never part of later loops' input. The final loop's branch is
``"#### <answer>"`` + EOS. The only channel between steps is therefore the
latent, which forces the loops to actually compute the intermediate results
instead of reading them from the context (see README Key Insights).

  loop 1: [ h0(Q) ; s1 ] -> CE(s1)   \  branches read Q's loop-i state,
  loop 2: [ h1(Q) ; s2 ] -> CE(s2)    | are dropped after the loss
  loop 3: [ h2(Q) ; s3 ] -> CE(s3)   /
  loop 4: [ h3(Q) ; "#### 405" ] -> CE(answer)

Steps are merged into num_loops-1 contiguous chunks when there are more steps
than intermediate loops; with fewer, leftover intermediate loops get the
answer text ("thought finished, hold the conclusion"). ``--num-loops 1`` is
the implicit no-looping control: question -> answer directly, same data.

Inference (see LlamaLoopedForCausalLM.generate_latent): loops 0..N-2 run once
over the question, then the answer is generated with single final-loop passes
— zero CoT tokens. NOTE: the standard generate()/lm-eval path does NOT match
this training scheme.

Run (single GPU — the custom loss path does not support DataParallel):
    CUDA_VISIBLE_DEVICES=6 python train/train_lora_llamalooped_latent.py \
        --model meta-llama/Llama-3.2-1B-Instruct [--per-loop-lora]

Loss is logged to Weights & Biases when USE_WANDB=True (pip install wandb).
"""

import os
import sys

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Make `models` importable when run as `python train/train_lora_llamalooped_latent.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llama_looped import LlamaLoopedForCausalLM

# Sibling import (train/ is on sys.path when this script is run directly;
# `train.common` would be shadowed by the top-level train.py).
from common import parse_args

# ---- config ----
RECUR_MODE = "latent"  # "latent" | "latent_renorm"
OUTPUT_DIR_BASE = "ckpts"
MAX_Q_LEN = 512  # question tokens
MAX_BRANCH_LEN = 128  # per-branch tokens
AUX_WEIGHT = 0.5  # weight of the intermediate-loop (step) CEs vs the answer CE

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

# Weights & Biases (pip install wandb; `wandb login` once).
USE_WANDB = True
WANDB_PROJECT = "llama_looped_lora"


def chunk_texts(steps, answer_text, n_slots):
    """Assign one branch text per intermediate loop slot.

    ``>= n_slots`` steps: merge into ``n_slots`` contiguous, near-even chunks.
    ``< n_slots`` steps: one step per slot, leftover slots get the answer text.
    """
    if n_slots <= 0:  # num_loops=1: no intermediate loops, implicit control
        return []
    k_steps = len(steps)
    if k_steps >= n_slots:
        size, rem = divmod(k_steps, n_slots)
        texts, start = [], 0
        for j in range(n_slots):
            end = start + size + (1 if j < rem else 0)
            texts.append("".join(s + " " for s in steps[start:end]))
            start = end
        return texts
    return [s + " " for s in steps] + [answer_text] * (n_slots - k_steps)


def build_dataset(tok, n_train=None, num_loops=4):
    ds = load_dataset("whynlp/gsm8k-aug", split="train")
    if n_train:
        ds = ds.select(range(n_train))

    def format_example(ex):
        # Input = chat-templated question ONLY; the reasoning steps never
        # appear in the input, they exist solely as per-loop branch targets.
        prompt_ids = tok.apply_chat_template(
            [{"role": "user", "content": ex["question"]}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )[:MAX_Q_LEN]
        answer_text = f"#### {ex['answer']}"
        branch_texts = chunk_texts(ex["steps"], answer_text, num_loops - 1)
        branch_texts.append(answer_text)  # final loop: the answer
        # print(branch_texts) # debug
        branch_ids = [
            tok(t, add_special_tokens=False)["input_ids"][:MAX_BRANCH_LEN]
            for t in branch_texts
        ]
        branch_ids[-1] = branch_ids[-1][: MAX_BRANCH_LEN - 1] + [tok.eos_token_id]
        return {
            "input_ids": prompt_ids,
            "attention_mask": [1] * len(prompt_ids),
            "branch_ids": branch_ids,
            "length": len(prompt_ids) + max(len(b) for b in branch_ids),
        }

    return ds.map(format_example, remove_columns=ds.column_names, num_proc=8)


def make_collator(pad_token_id, num_loops):
    def collate(features):
        q_max = max(len(f["input_ids"]) for f in features)
        b_max = max(len(b) for f in features for b in f["branch_ids"])
        input_ids, attention_mask = [], []
        branch_ids = torch.full(
            (len(features), num_loops, b_max), pad_token_id, dtype=torch.long
        )
        branch_mask = torch.zeros(len(features), num_loops, b_max, dtype=torch.long)
        for k, f in enumerate(features):
            pad = q_max - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_token_id] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
            for i, b in enumerate(f["branch_ids"]):
                branch_ids[k, i, : len(b)] = torch.tensor(b, dtype=torch.long)
                branch_mask[k, i, : len(b)] = 1
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "branch_ids": branch_ids,
            "branch_mask": branch_mask,
        }

    return collate


class LatentBranchTrainer(Trainer):
    """Per-loop branch CE on the readouts of forward_latent_branches.

    Calls the unwrapped looped model directly (bypasses the Trainer's model
    wrapper), so this path is single-GPU / DDP-per-process only — no
    DataParallel."""

    def __init__(self, *args, looped_lm=None, aux_weight=AUX_WEIGHT, **kwargs):
        super().__init__(*args, **kwargs)
        self.looped_lm = looped_lm  # unwrapped LlamaLoopedForCausalLM
        self.aux_weight = aux_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        branch_ids = inputs["branch_ids"]  # (B, num_loops, Lb)
        branch_mask = inputs["branch_mask"].bool()
        device_type = branch_ids.device.type
        # The custom entry point is not wrapped by accelerate, so apply the
        # bf16 autocast ourselves (keeps fp32 LoRA params from upcasting the
        # bf16 residual stream).
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=self.args.bf16):
            readouts = self.looped_lm.forward_latent_branches(
                inputs["input_ids"],
                inputs["attention_mask"],
                branch_ids,
                branch_mask.long(),
            )
        lm_head = self.looped_lm.lm_head
        last = len(readouts) - 1
        loss, aux = None, []
        for i, pred in enumerate(readouts):
            # pred position j predicts branch token j (no shift needed:
            # position 0 already reads from the last question token).
            msk = branch_mask[:, i]
            logits = lm_head(pred[msk].to(lm_head.weight.dtype))
            ce = F.cross_entropy(logits.float(), branch_ids[:, i][msk])
            if i == last:
                loss = ce
            else:
                aux.append(ce)
        if aux:
            loss = loss + self.aux_weight * torch.stack(aux).mean()
        return (loss, readouts) if return_outputs else loss


def main():
    args_cli = parse_args(__doc__)
    model_name = args_cli.model
    model_short = model_name.rstrip("/").split("/")[-1]
    per_loop_lora = args_cli.per_loop_lora
    num_loops = args_cli.num_loops
    # Same suffix for output_dir and run_name, so runs with different settings
    # never overwrite each other's checkpoints.
    suffix = (f"_L{num_loops}" if num_loops != 4 else "") + (
        "_perloop" if per_loop_lora else ""
    )
    output_dir = os.path.join(
        OUTPUT_DIR_BASE, f"{model_short}_looped_lora_latent{suffix}"
    )
    run_name = (
        f"{model_short}-gsm8kaug-latent-lora-{RECUR_MODE}-L{num_loops}-r{LORA_R}"
        + suffix.replace("_", "-")
    )

    if USE_WANDB:
        os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # right-pad for causal-LM training

    train_ds = build_dataset(tok, args_cli.n_train, num_loops)
    # print(train_ds[0])  # debug

    model = LlamaLoopedForCausalLM.from_pretrained(
        model_name,
        num_loops=num_loops,
        recur_mode=RECUR_MODE,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False  # required with gradient checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # needed for grad-ckpt + LoRA
    looped_lm = model

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
        remove_unused_columns=False,  # keep branch_ids/branch_mask for the collator
        label_names=["branch_ids"],
        # Batch similar lengths together (less padding); uses the "length" column.
        train_sampling_strategy="group_by_length",
    )

    trainer = LatentBranchTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=make_collator(tok.pad_token_id, num_loops),
        looped_lm=looped_lm,
    )
    trainer.train()

    adapter_dir = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    print(f"saved LoRA adapter to: {adapter_dir}")
    load_hint = (
        "load with:\n"
        f"  base = LlamaLoopedForCausalLM.from_pretrained('{model_name}',"
        f" num_loops={num_loops}, recur_mode='{RECUR_MODE}', dtype='bfloat16')\n"
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
        load_hint += f"  model.get_base_model().set_loop_adapters({adapter_names!r})\n"
    else:
        load_hint += f"  model = PeftModel.from_pretrained(base, '{adapter_dir}')\n"
    load_hint += (
        "generate with (NOT the standard generate()):\n"
        "  out = model.get_base_model().generate_latent(input_ids,"
        " max_new_tokens=32, eos_token_id=tok.eos_token_id)"
    )
    print(load_hint)


if __name__ == "__main__":
    main()
