"""
PoC: loopformer vs loopformer+damping on HellaSwag / PIQA / WinoGrande / ARC
Tests loop counts: 4 (training-time) and 8 (2x test-time scaling)

Usage:
    python eval_hellaswag_poc.py \
        [--task hellaswag|piqa|winogrande|arc] \
        [--loops 4 8] [--num_examples 1000] [--device cuda]

Dependencies:  pip install datasets tiktoken
"""
import argparse
import torch
import torch.nn.functional as F
from dataclasses import asdict
from datasets import load_dataset
from tqdm import tqdm

from models.loopformer import GPTConfig, GPT


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: str, use_damping: bool = False) -> GPT:
    """Load model from either .pt checkpoint or HuggingFace model ID."""
    if ckpt_path.endswith('.pt'):
        # local checkpoint
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg_obj = ckpt['config']
        cfg_dict = asdict(cfg_obj) if hasattr(cfg_obj, '__dataclass_fields__') else dict(cfg_obj)
        model = GPT(GPTConfig(**cfg_dict))
        sd = {k.removeprefix('_orig_mod.'): v for k, v in ckpt['model'].items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  [warn] missing keys in {ckpt_path}: {missing}")
    else:
        # HuggingFace model ID
        cfg = GPTConfig(use_damping=use_damping)
        model = GPT.from_pretrained(ckpt_path, config=cfg)
    model.eval().to(device)
    return model


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def log_likelihood(model: GPT, ctx_ids: list, cont_ids: list,
                   steps: list, device: str, dtype: torch.dtype) -> float:
    """Sum log-prob of cont_ids tokens given ctx_ids prefix."""
    ids = torch.tensor(ctx_ids + cont_ids, dtype=torch.long, device=device).unsqueeze(0)

    # pass targets=-1 (all ignored) to force model to return full-sequence logits
    dummy = torch.full_like(ids, -1)
    with torch.amp.autocast(device_type=device.split(':')[0], dtype=dtype):
        logits, _, _ = model(ids, targets=dummy, steps=steps)

    log_probs = F.log_softmax(logits[0], dim=-1)   # (T, vocab)
    cont_start = len(ctx_ids)
    targets = torch.tensor(cont_ids, dtype=torch.long, device=device)
    ll = log_probs[cont_start - 1: cont_start - 1 + len(cont_ids), targets].diagonal().sum().item()
    return ll


# ---------------------------------------------------------------------------
# Task evaluation
# ---------------------------------------------------------------------------

def evaluate_hellaswag(model: GPT, dataset, num_loops: int,
                       device: str, dtype: torch.dtype, num_examples: int) -> float:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    steps = [1 / num_loops] * num_loops

    correct = 0
    for ex in tqdm(dataset.select(range(num_examples)), desc=f"loops={num_loops}", leave=False):
        ctx_ids = enc.encode(ex['ctx'])
        label   = int(ex['label'])
        scores  = [
            log_likelihood(model, ctx_ids, enc.encode(" " + end),
                           steps, device, dtype)
            for end in ex['endings']
        ]
        if scores.index(max(scores)) == label:
            correct += 1

    return correct / num_examples


def evaluate_piqa(model: GPT, dataset, num_loops: int,
                  device: str, dtype: torch.dtype, num_examples: int) -> float:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    steps = [1 / num_loops] * num_loops

    correct = 0
    for ex in tqdm(dataset.select(range(num_examples)), desc=f"loops={num_loops}", leave=False):
        goal_ids = enc.encode(ex['goal'])
        label = int(ex['label'])
        scores = [
            log_likelihood(model, goal_ids, enc.encode(" " + sol),
                           steps, device, dtype)
            for sol in [ex['sol1'], ex['sol2']]
        ]
        if scores.index(max(scores)) == label:
            correct += 1

    return correct / num_examples


def evaluate_winogrande(model: GPT, dataset, num_loops: int,
                        device: str, dtype: torch.dtype, num_examples: int) -> float:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    steps = [1 / num_loops] * num_loops

    correct = 0
    for ex in tqdm(dataset.select(range(num_examples)), desc=f"loops={num_loops}", leave=False):
        sentence_ids = enc.encode(ex['sentence'])
        label = 0 if ex['answer'] == '1' else 1
        scores = [
            log_likelihood(model, sentence_ids, enc.encode(" " + opt),
                           steps, device, dtype)
            for opt in [ex['option1'], ex['option2']]
        ]
        if scores.index(max(scores)) == label:
            correct += 1

    return correct / num_examples


def evaluate_arc(model: GPT, dataset, num_loops: int,
                 device: str, dtype: torch.dtype, num_examples: int) -> float:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    steps = [1 / num_loops] * num_loops

    correct = 0
    for ex in tqdm(dataset.select(range(num_examples)), desc=f"loops={num_loops}", leave=False):
        question_ids = enc.encode(ex['question'])
        label = ord(ex['answerKey']) - ord('A')  # 'A'->0, 'B'->1, etc.
        scores = [
            log_likelihood(model, question_ids, enc.encode(" " + choice),
                           steps, device, dtype)
            for choice in ex['choices']['text']
        ]
        if scores.index(max(scores)) == label:
            correct += 1

    return correct / num_examples


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',         default='hellaswag', choices=['hellaswag', 'piqa', 'winogrande', 'arc-easy', 'arc-challenge'])
    parser.add_argument('--loops',        nargs='+', type=int, default=[4, 8])
    parser.add_argument('--num_examples', type=int,  default=1000)
    parser.add_argument('--device',       default='cuda')
    args = parser.parse_args()

    dtype  = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    device = args.device

    # load dataset
    if args.task == 'hellaswag':
        print("Loading HellaSwag validation split...")
        ds = load_dataset("Rowan/hellaswag", split="validation")
        eval_fn = evaluate_hellaswag
    elif args.task == 'piqa':
        print("Loading PIQA validation split...")
        ds = load_dataset("ybisk/piqa", split="validation")
        eval_fn = evaluate_piqa
    elif args.task == 'winogrande':
        print("Loading WinoGrande validation split...")
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
        eval_fn = evaluate_winogrande
    elif args.task == 'arc-easy':
        print("Loading ARC-Easy validation split...")
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
        eval_fn = evaluate_arc
    elif args.task == 'arc-challenge':
        print("Loading ARC-Challenge validation split...")
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test") # validaton only 299 rows, so use test split instead
        eval_fn = evaluate_arc
    else:
        raise ValueError(f"Unknown task: {args.task}")

    ckpt_base = "armenjeddi/LoopFormer-3block-8iterations"
    ckpt_damping = "runs/loopformer_damping/ckpt.pt"
    ckpt_wo_damping = "runs/loopformer_wo_damping/ckpt.pt"

    checkpoints = {
        'loopformer':         ckpt_base,
        'loopformer+damping': ckpt_damping,
        'loopformer+wo_damping': ckpt_wo_damping,
    }

    results = {}
    for name, path in checkpoints.items():
        print(f"\nLoading {name} from {path}")
        use_damping = (name == 'loopformer+damping')
        model = load_model(path, device, use_damping=use_damping)
        for num_loops in args.loops:
            acc = eval_fn(model, ds, num_loops, device, dtype, args.num_examples)
            results[(name, num_loops)] = acc
        del model
        torch.cuda.empty_cache()

    # print table (rows = models, cols = loop counts)
    print("\n" + args.task.upper())
    header = "\t" + "\t".join(str(l) for l in args.loops)
    print(header)
    for name in checkpoints:
        row = name
        for num_loops in args.loops:
            acc = results[(name, num_loops)]
            row += f"\t{acc:.4f}"
        print(row)

    print("\nTest-time scaling delta (max_loops - min_loops):")
    lo, hi = min(args.loops), max(args.loops)
    for name in checkpoints:
        delta = results[(name, hi)] - results[(name, lo)]
        print(f"  {name}: {delta:+.4f}  ({lo} loops → {hi} loops)")


if __name__ == '__main__':
    main()
