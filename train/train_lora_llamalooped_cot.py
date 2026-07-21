"""LoRA SFT of the looped Llama on GSM8K-Aug with per-loop CoT deep supervision.

Uses whynlp/gsm8k-aug, whose answers are split into calculator steps
(``['<<5*3=15>>', '<<15*2=30>>', ...]`` + final answer). The training sequence
is the ordinary teacher-forced ``question + steps + "#### answer"``, but each
loop gets its own loss mask so that loop depth aligns with reasoning depth:

  loop i   (i < N-1): CE only on the tokens of step-chunk i
  loop N-1 (last):    CE on the full response (all steps + answer)

Examples with more than N-1 steps have their steps merged into N-1 contiguous
chunks; with fewer, the leftover intermediate loops supervise the answer span
("the thought is finished, hold the conclusion"). Intermediate loop hidden
states are read out through the frozen final norm + lm_head (logit-lens style);
the total loss is ``CE_last + AUX_WEIGHT * mean(CE_intermediate)``. Inference
is unchanged (generate with the last loop), so results are directly comparable
to train_lora_llamalooped.py.

Run (single GPU):
    CUDA_VISIBLE_DEVICES=6 python train/train_lora_llamalooped_cot.py \
        --model meta-llama/Llama-3.2-3B-Instruct [--per-loop-lora]

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

# Make `models` importable when run as `python train/train_lora_llamalooped_cot.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llama_looped import LlamaLoopedForCausalLM

# Sibling import (train/ is on sys.path when this script is run directly;
# `train.common` would be shadowed by the top-level train.py).
from common import parse_args, prc_skip_layers

# ---- config ----
RECUR_MODE = "latent"  # "latent" | "latent_renorm" | "token" | "soft"
SKIP_LAYERS = None
OUTPUT_DIR_BASE = "ckpts"  # actual dir: {OUTPUT_DIR_BASE}/{model_short}_looped_lora_cot
MAX_LEN = 1024
AUX_WEIGHT = 0.5  # weight of the intermediate-loop losses relative to the last loop

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


def chunk_spans(step_spans, ans_span, n_slots):
    """Assign one token span per intermediate loop slot.

    ``>= n_slots`` steps: merge into ``n_slots`` contiguous, near-even chunks.
    ``< n_slots`` steps: one step per slot, leftover slots get the answer span.
    """
    # print(f"step_spans: {step_spans}, ans_span: {ans_span}, n_slots: {n_slots}")    # debug
    if n_slots <= 0:  # num_loops=1: no intermediate loops, plain SFT
        return []
    k_steps = len(step_spans)
    if k_steps >= n_slots:
        size, rem = divmod(k_steps, n_slots)
        spans, start = [], 0
        for j in range(n_slots):
            end = start + size + (1 if j < rem else 0)
            spans.append((step_spans[start][0], step_spans[end - 1][1]))
            start = end
        return spans
    return list(step_spans) + [ans_span] * (n_slots - k_steps)


def build_dataset(tok, n_train=None, num_loops=4):
    ds = load_dataset("whynlp/gsm8k-aug", split="train")
    # ds = load_dataset("whynlp/gsm8k-aug", split="train[:30]")    # debug
    if n_train:
        ds = ds.select(range(n_train))

    def format_example(ex):
        # Prompt = chat-templated user turn (masked in all loops). Response
        # pieces are tokenized separately so span boundaries are exact.
        prompt_ids = tok.apply_chat_template(
            [{"role": "user", "content": ex["question"]}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        piece_texts = [s + " " for s in ex["steps"]] + [f"#### {ex['answer']}"]
        piece_ids = [tok(t, add_special_tokens=False)["input_ids"] for t in piece_texts]
        piece_ids[-1] = piece_ids[-1] + [tok.eos_token_id]

        input_ids = list(prompt_ids)
        spans = []
        for ids in piece_ids:
            spans.append((len(input_ids), len(input_ids) + len(ids)))
            input_ids.extend(ids)
        input_ids = input_ids[:MAX_LEN]
        seq_len = len(input_ids)

        slot_spans = chunk_spans(spans[:-1], spans[-1], num_loops - 1)
        loop_labels = [[-100] * seq_len for _ in range(num_loops)]
        for j, (a, b) in enumerate(slot_spans):
            loop_labels[j][a:min(b, seq_len)] = input_ids[a:min(b, seq_len)]
        # Last loop: full response, so autoregressive generation stays intact.
        resp_start = min(len(prompt_ids), seq_len)
        loop_labels[-1][resp_start:] = input_ids[resp_start:]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * seq_len,
            "loop_labels": loop_labels,
            "length": seq_len,  # for group_by_length (LengthGroupedSampler)
            # First token of the answer span ("#### <answer>"), for weighting
            # the last loop's CE (step region vs answer region).
            "answer_start": min(spans[-1][0], seq_len),
        }
    # print(ds[0])   # debug
    ds = ds.map(format_example, remove_columns=ds.column_names, num_proc=8)
    # print(ds[0])   # debug
    # print(len(ds[0]['loop_labels']))   # debug
    # Drop examples whose response was fully truncated away (no final-loop loss).
    return ds.filter(
        lambda ex: any(t != -100 for t in ex["loop_labels"][-1]), num_proc=8
    )


def make_collator(pad_token_id):
    def collate(features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, loop_labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_token_id] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
            loop_labels.append([row + [-100] * pad for row in f["loop_labels"]])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "loop_labels": torch.tensor(loop_labels, dtype=torch.long),
            "answer_start": torch.tensor(
                [f["answer_start"] for f in features], dtype=torch.long
            ),
        }

    return collate


class PerLoopTrainer(Trainer):
    """Reads per-loop hidden states off the looped model and applies the
    per-loop loss masks; the model's own lm_head/loss path is bypassed."""

    def __init__(
        self,
        *args,
        looped_lm=None,
        aux_weight=AUX_WEIGHT,
        final_step_weight=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.looped_lm = looped_lm  # unwrapped LlamaLoopedForCausalLM
        self.aux_weight = aux_weight
        # Weight of step-region tokens in the last loop's CE (answer region is
        # always 1.0). < 1 shifts the last loop's gradient toward the
        # conclusion, leaving step generation mostly to the intermediate loops.
        self.final_step_weight = final_step_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loop_labels = inputs["loop_labels"]  # (B, NUM_LOOPS, L)
        answer_start = inputs["answer_start"]  # (B,)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            logits_to_keep=1,  # full-vocab logits are computed below instead
        )
        # Per-loop normed hiddens, NUM_LOOPS x (B, L, H); read from the output
        # (not looped_lm.model.loop_hiddens) so this also works under DataParallel.
        hiddens = outputs.hidden_states
        lm_head = self.looped_lm.lm_head
        last = len(hiddens) - 1
        loss, aux = None, []
        for i, h in enumerate(hiddens):
            labels = loop_labels[:, i, 1:]  # logits at t predict token t+1
            mask = labels != -100
            if not mask.any():
                continue  # span fully truncated away
            # accelerate's mixed-precision wrapper converts model outputs to
            # fp32; cast back since lm_head runs outside the autocast ctx.
            logits = lm_head(h[:, :-1][mask].to(lm_head.weight.dtype))
            if i == last and self.final_step_weight != 1.0:
                # Weighted mean: step tokens x final_step_weight, answer x 1.0.
                # Label index j scores the token at position j + 1.
                pos = torch.arange(1, labels.shape[1] + 1, device=labels.device)
                weights = torch.where(
                    pos.unsqueeze(0) >= answer_start.unsqueeze(1),
                    1.0,
                    self.final_step_weight,
                )[mask]
                ce_tok = F.cross_entropy(
                    logits.float(), labels[mask], reduction="none"
                )
                ce = (ce_tok * weights).sum() / weights.sum()
            else:
                ce = F.cross_entropy(logits.float(), labels[mask])
            if i == last:
                loss = ce
            else:
                aux.append(ce)
        if aux:
            loss = loss + self.aux_weight * torch.stack(aux).mean()
        return (loss, outputs) if return_outputs else loss


def main():
    args_cli = parse_args(__doc__)
    model_name = args_cli.model
    model_short = model_name.rstrip("/").split("/")[-1]
    per_loop_lora = args_cli.per_loop_lora
    num_loops = args_cli.num_loops
    prc = args_cli.prc  # None or [prelude, coda]
    # Same suffix for output_dir and run_name, so runs with different settings
    # never overwrite each other's checkpoints.
    suffix = (
        (f"_L{num_loops}" if num_loops != 4 else "")
        + ("_perloop" if per_loop_lora else "")
        + (
            f"_fsw{args_cli.final_step_weight}"
            if args_cli.final_step_weight != 1.0
            else ""
        )
        + (f"_prc{prc[0]}-{prc[1]}" if prc else "")
    )
    output_dir = os.path.join(OUTPUT_DIR_BASE, f"{model_short}_looped_lora_cot{suffix}")
    run_name = (
        f"{model_short}-gsm8kaug-cot-lora-{RECUR_MODE}-L{num_loops}-r{LORA_R}"
        + suffix.replace("_", "-")
    )

    if USE_WANDB:
        os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # right-pad for causal-LM training

    train_ds = build_dataset(tok, args_cli.n_train, num_loops)

    model = LlamaLoopedForCausalLM.from_pretrained(
        model_name,
        num_loops=num_loops,
        recur_mode=RECUR_MODE,
        skip_layers=SKIP_LAYERS,
        dtype=torch.bfloat16,
    )
    if prc:
        skip = prc_skip_layers(
            prc[0], prc[1], model.config.num_hidden_layers, num_loops
        )
        model.set_skip_layers(skip)
        print(f"[INFO] P-R-C skip_layers: {skip}")
    model.config.use_cache = False  # required with gradient checkpointing
    use_gradient_checkpointing = True  # True for memory savings, but slower
    if use_gradient_checkpointing == True:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # needed for grad-ckpt + LoRA
    model.model.collect_loop_hiddens = True  # expose per-loop hiddens for the loss
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
        remove_unused_columns=False,  # keep loop_labels for the collator
        label_names=["loop_labels"],
        # Batch similar lengths together (less padding); uses the "length" column.
        train_sampling_strategy="group_by_length",
    )

    trainer = PerLoopTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=make_collator(tok.pad_token_id),
        looped_lm=looped_lm,
        final_step_weight=args_cli.final_step_weight,
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
        load_hint += f"  model.get_base_model().set_loop_adapters({adapter_names!r})"
    else:
        load_hint += f"  model = PeftModel.from_pretrained(base, '{adapter_dir}')"
    print(load_hint)


if __name__ == "__main__":
    main()
