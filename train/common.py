"""Shared CLI for the looped-LoRA training scripts."""

import argparse


def prc_skip_layers(prelude, coda, num_layers, num_loops):
    """skip_layers dict realizing P-R^N-C with the existing skip mechanism.

    Loop 0 runs prelude+core (skips coda), middle loops run core only, the
    last loop runs core+coda (skips prelude). Returns None when nothing is
    skipped (e.g. num_loops=1: the full stack runs once).
    """
    assert prelude >= 0 and coda >= 0 and prelude + coda < num_layers, (
        f"invalid P-R-C split: prelude={prelude}, coda={coda}, "
        f"num_layers={num_layers}"
    )
    pre = list(range(prelude))
    cod = list(range(num_layers - coda, num_layers))
    skip = {}
    for i in range(num_loops):
        s = []
        if i > 0:
            s += pre
        if i < num_loops - 1:
            s += cod
        if s:
            skip[i] = s
    return skip or None


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
        "--num-loops",
        type=int,
        default=4,
        help="number of times the decoder stack is looped "
        "(1 = plain non-looped model, e.g. as a no-looping control)",
    )
    parser.add_argument(
        "--final-step-weight",
        type=float,
        default=1.0,
        help="(CoT script only) weight of the step-region tokens in the last "
        "loop's CE loss; the answer region always has weight 1.0",
    )
    parser.add_argument(
        "--prc",
        type=int,
        nargs=2,
        metavar=("PRELUDE", "CODA"),
        default=None,
        help="Prelude-Recurrent-Coda looping: the first PRELUDE and last CODA "
        "layers run once (in the first/last loop); only the middle block "
        "loops. Omit to loop the full stack",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="train on only the first N examples, e.g. 2000 for a quick run "
        "(default: full train split)",
    )
    return parser.parse_args()
