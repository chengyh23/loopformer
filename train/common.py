"""Shared CLI for the looped-LoRA training scripts."""

import argparse


def parse_args(description=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HF model id or local path of the base model to loop and fine-tune",
    )
    parser.add_argument(
        "--per-loop-lora",
        action="store_true",
        help="train a separate LoRA adapter for each loop instead of one "
        "adapter shared across all loops",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="per-device train batch size",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=16,
        help="gradient accumulation steps (effective batch = "
        "batch-size * grad-accum * n_gpus)",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="train on only the first N examples, e.g. 2000 for a quick run "
        "(default: full train split)",
    )
    return parser.parse_args()
