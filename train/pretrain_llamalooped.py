"""Pretrain a looped Llama from scratch on the SmolLM2-135M backbone.

Builds a ``LlamaLoopedForCausalLM`` whose architecture (layers/width/vocab/rope)
is inherited from ``HuggingFaceTB/SmolLM2-135M`` but whose weights are randomly
initialized (``--init-from scratch``, default) or loaded from that checkpoint
(``--init-from backbone``). Trains a causal-LM objective on a streaming subset
of FineWeb-Edu (default the 10B-token ``sample-10BT`` config) for a fixed token
budget. Full-parameter training (no LoRA); num_loops / sandwich-norm / AttnRes
are all trained end-to-end and saved with the model.

Data is streamed, tokenized, and packed into fixed-length blocks (no padding,
no loss masking: labels == input_ids). For multi-GPU DDP the stream is sharded
by rank via ``split_dataset_by_node`` and batch dispatch is disabled so each
rank reads its own shard.

Run (multi-GPU DDP, N GPUs):
    torchrun --nproc_per_node=N train/pretrain_llamalooped.py \
        --num-loops 4 --recur-mode latent \
        --per-device-batch-size 16 --grad-accum 8 --seq-len 2048 \
        --tokens 10_000_000_000

Run (single GPU smoke test, tiny budget):
    CUDA_VISIBLE_DEVICES=0 python train/pretrain_llamalooped.py \
        --tokens 2_000_000 --save-steps 50 --output-dir ckpts/smoke

Loss is logged to Weights & Biases when --wandb is passed (pip install wandb).
"""

import argparse
import math
import os
import sys

import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# Make `models` importable when run as `python train/pretrain_llamalooped.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llama_looped import LlamaLoopedForCausalLM
from models.llama_looped.configuration_llama_looped import LlamaLoopedConfig

# Sibling import (train/ is on sys.path when run directly; `train.common` would
# be shadowed by the top-level train.py).
from common import prc_skip_layers


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    # architecture / init
    p.add_argument("--backbone", default="HuggingFaceTB/SmolLM2-135M",
                   help="source of the architecture config and tokenizer")
    p.add_argument("--init-from", choices=["scratch", "backbone"], default="scratch",
                   help="scratch: random init on the backbone's architecture; "
                        "backbone: load the backbone's pretrained weights")
    # loop config
    p.add_argument("--num-loops", type=int, default=4)
    p.add_argument("--recur-mode", default="latent",
                   choices=["latent", "latent_renorm", "token", "soft"])
    p.add_argument("--sandwich-norm", action="store_true",
                   help="use the 4-norm sandwich decoder-layer layout")
    p.add_argument("--loop-attn-res", action="store_true",
                   help="Attention Residuals over the unrolled loop-depth")
    p.add_argument("--prc", type=int, nargs=2, metavar=("PRELUDE", "CODA"),
                   default=None,
                   help="Prelude-Recurrent-Coda: first PRELUDE / last CODA "
                        "layers run once; only the middle block loops")
    # data
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--text-column", default="text")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--shuffle-buffer", type=int, default=10000)
    # optimization
    p.add_argument("--tokens", type=int, default=10_000_000_000,
                   help="total training-token budget; sets max_steps")
    p.add_argument("--per-device-batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4,
                   help="peak LR (small models often tolerate up to ~3e-3)")
    p.add_argument("--warmup-ratio", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false",
                   help="disable gradient checkpointing (uses more memory)")
    # io / logging
    p.add_argument("--output-dir", default=None)
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="llama_looped_pretrain")
    p.add_argument(
        "--resume", nargs="?", const=True, default=False,
        help="resume training: bare --resume picks the latest checkpoint in "
        "--output-dir; --resume PATH resumes from that checkpoint dir. Re-run "
        "with the SAME args (num-loops/recur-mode/backbone/batch/grad-accum/"
        "tokens) so max_steps and the LR schedule match the saved state.",
    )
    return p.parse_args()


def build_model(args):
    """Looped model on the backbone architecture; random or backbone-init."""
    overrides = dict(
        num_loops=args.num_loops,
        recur_mode=args.recur_mode,
        use_loop_sandwichnorm=args.sandwich_norm,
        use_loop_attn_residual=args.loop_attn_res,
    )
    if args.init_from == "backbone":
        model = LlamaLoopedForCausalLM.from_pretrained(
            args.backbone, dtype=torch.bfloat16, **overrides
        )
    else:
        # Inherit dims/vocab/rope from the backbone config, random weights.
        base = AutoConfig.from_pretrained(args.backbone)
        base_dict = base.to_dict()
        # Drop identity/meta keys so they don't clobber the looped config's
        # model_type / architecture registration.
        for k in ("model_type", "architectures", "transformers_version",
                  "_name_or_path", "torch_dtype", "auto_map"):
            base_dict.pop(k, None)
        config = LlamaLoopedConfig(**{**base_dict, **overrides})
        model = LlamaLoopedForCausalLM(config).to(torch.bfloat16)
    if args.prc:
        skip = prc_skip_layers(
            args.prc[0], args.prc[1], model.config.num_hidden_layers, args.num_loops
        )
        model.set_skip_layers(skip)
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"[INFO] P-R-C skip_layers: {skip}")
    model.config.use_cache = False  # required with gradient checkpointing
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    return model


def build_dataset(args, tok, rank, world_size):
    """Streaming FineWeb-Edu, tokenized and packed into seq_len blocks."""
    ds = load_dataset(
        args.dataset, args.dataset_config, split="train", streaming=True
    )
    if world_size > 1:
        # Each rank reads a disjoint shard (batch dispatch is disabled below).
        ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    eos = tok.eos_token_id
    text_col = args.text_column

    def tokenize(batch):
        out = tok(batch[text_col], add_special_tokens=False)
        for ids in out["input_ids"]:
            ids.append(eos)  # document separator
        return {"input_ids": out["input_ids"]}

    ds = ds.map(tokenize, batched=True, remove_columns=list(ds.column_names or []))

    seq_len = args.seq_len

    def group(batch):
        concat = [t for ids in batch["input_ids"] for t in ids]
        total = (len(concat) // seq_len) * seq_len
        blocks = [concat[i:i + seq_len] for i in range(0, total, seq_len)]
        return {"input_ids": blocks, "labels": [b[:] for b in blocks]}

    ds = ds.map(group, batched=True, remove_columns=["input_ids"])
    return ds


def main():
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    is_main = rank == 0

    if args.wandb and is_main:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    report_to = ["wandb"] if args.wandb else "none"

    output_dir = args.output_dir or (
        f"ckpts/looped{args.num_loops}_{args.recur_mode}"
        f"{'_sw' if args.sandwich_norm else ''}"
        f"{'_ar' if args.loop_attn_res else ''}_fineweb"
    )

    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = build_model(args)
    train_ds = build_dataset(args, tok, rank, world_size)

    # token budget -> max_steps. Effective tokens/step spans all ranks.
    tokens_per_step = (
        args.per_device_batch_size * args.seq_len * args.grad_accum * world_size
    )
    max_steps = max(1, args.tokens // tokens_per_step)
    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[INFO] params={n_params/1e6:.1f}M | world_size={world_size} | "
              f"tokens/step={tokens_per_step:,} | max_steps={max_steps:,} | "
              f"init={args.init_from} | out={output_dir}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=report_to,
        run_name=os.path.basename(output_dir),
        seed=args.seed,
        dataloader_num_workers=2,
        # Streaming + DDP: we sharded per-rank above, so each process must read
        # its own shard rather than have the main process dispatch batches.
        accelerator_config={"dispatch_batches": False},
        # On resume, don't replay-and-discard already-seen batches: the dataset
        # is a streaming IterableDataset, so skipping would re-tokenize the whole
        # stream (pathologically slow). Model/optimizer/LR/step-counter are still
        # restored; only the data stream restarts from the top of the shuffle.
        ignore_data_skip=True,
        # gradient_checkpointing is toggled on the model directly (build_model).
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=default_data_collator,
    )
    if args.resume and is_main:
        print(f"[INFO] resuming from checkpoint: {args.resume}")
    trainer.train(resume_from_checkpoint=args.resume)

    if is_main:
        trainer.save_model(output_dir)
        tok.save_pretrained(output_dir)
        print(f"[INFO] saved pretrained looped model to: {output_dir}")


if __name__ == "__main__":
    main()
