"""
PoC: loopformer vs loopformer+damping on HellaSwag / PIQA / WinoGrande / ARC

Usage:
    python eval_poc.py \
        [--task hellaswag|piqa|winogrande|arc-easy|arc-challenge] \
        [--loops 4 8 12 16 20 24] [--num_examples 1000] [--device cuda]

Dependencies:  pip install datasets tiktoken
"""
import argparse
import torch
import torch.nn.functional as F
from dataclasses import asdict
from datasets import load_dataset
from tqdm import tqdm
import csv
from datetime import datetime

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
        missing, _ = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  [warn] missing keys in {ckpt_path}: {missing}")
    else:
        # HuggingFace model ID
        cfg = GPTConfig(use_damping=use_damping)
        model = GPT.from_pretrained(ckpt_path, config=cfg)
    model.eval().to(device)

    # print model config
    print(f"  Model config: {asdict(model.config)}")
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
def load_task(current_task: str):
    # load dataset for current task
    if current_task == 'hellaswag':
        print("Loading HellaSwag validation split...")
        ds = load_dataset("Rowan/hellaswag", split="validation")
        eval_fn = evaluate_hellaswag
    elif current_task == 'piqa':
        print("Loading PIQA validation split...")
        ds = load_dataset("ybisk/piqa", split="validation")
        eval_fn = evaluate_piqa
    elif current_task == 'winogrande':
        print("Loading WinoGrande validation split...")
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
        eval_fn = evaluate_winogrande
    elif current_task == 'arc-easy':
        print("Loading ARC-Easy validation split...")
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
        eval_fn = evaluate_arc
    elif current_task == 'arc-challenge':
        print("Loading ARC-Challenge validation split...")
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")
        eval_fn = evaluate_arc
    elif current_task == 'gsm8k':
        print("Loading GSM8K validation split...")
        ds = load_dataset("openai/gsm8k", "main", split="test")  # test split has ground truth
        eval_fn = evaluate_gsm8k
    else:
        raise ValueError(f"Unknown task: {current_task}")
    return ds, eval_fn

def evaluate_hellaswag(model: GPT, dataset, num_loops: int,
                       device: str, dtype: torch.dtype, num_examples: int) -> float:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    steps = [1 / num_loops] * num_loops

    if num_examples is None:
        num_examples = len(dataset)

    correct = 0
    for ex in tqdm(dataset.select(range(min(num_examples, len(dataset)))), desc=f"loops={num_loops}", leave=False):
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

    if num_examples is None:
        num_examples = len(dataset)

    correct = 0
    for ex in tqdm(dataset.select(range(min(num_examples, len(dataset)))), desc=f"loops={num_loops}", leave=False):
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

    if num_examples is None:
        num_examples = len(dataset)

    correct = 0
    for ex in tqdm(dataset.select(range(min(num_examples, len(dataset)))), desc=f"loops={num_loops}", leave=False):
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

    if num_examples is None:
        num_examples = len(dataset)

    correct = 0
    for ex in tqdm(dataset.select(range(min(num_examples, len(dataset)))), desc=f"loops={num_loops}", leave=False):
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


def evaluate_gsm8k(model: GPT, dataset, num_loops: int,
                   device: str, dtype: torch.dtype, num_examples: int) -> float:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    steps = [1 / num_loops] * num_loops

    if num_examples is None:
        num_examples = len(dataset)

    correct = 0
    for ex in tqdm(dataset.select(range(min(num_examples, len(dataset)))), desc=f"loops={num_loops}", leave=False):
        question_ids = enc.encode(ex['question'])
        # extract answer from "#### <answer>" format
        answer_text = ex['answer'].split('####')[-1].strip()
        answer_ids = enc.encode(" " + answer_text)

        ll = log_likelihood(model, question_ids, answer_ids, steps, device, dtype)
        # simple heuristic: if log likelihood is not too negative, count as correct
        # (this is a rough approximation; ideally you'd parse the model's generated text)
        if ll > -50:  # threshold, can be tuned
            correct += 1

    return correct / num_examples


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',         nargs='*',
                        choices=['hellaswag', 'piqa', 'winogrande', 'arc-easy', 'arc-challenge', 'gsm8k'],
                        help='task(s) to evaluate. if not specified, eval all tasks')
    parser.add_argument('--loops',        nargs='+', type=int, default=[4, 8, 12, 16, 20, 24])
    parser.add_argument('--num_examples', type=int,  default=None, help='number of examples to evaluate. if not specified, use full dataset')
    parser.add_argument('--device',       default='cuda')
    args = parser.parse_args()

    # default to all tasks if none specified
    if not args.task:
        args.task = ['hellaswag', 'piqa', 'winogrande', 'arc-easy', 'arc-challenge', 'gsm8k']

    dtype  = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    device = args.device

    # checkpoint configs
    test_configs = [
        # name, ckpt_path, use_damping
        # ('loopformer', "armenjeddi/LoopFormer-3block-8iterations", False),
        ('loopformer_fineweb', "armenjeddi/LoopFormer-3block-8iterations-FineWeb300K", False),
        # ('loopformer+damping', "runs/loopformer_damping/ckpt.pt", True),
    ]

    # collect all results: (config_name, task) -> {num_loops: acc}
    all_results = {}


    # evaluate each config
    for name, ckpt_path, use_damping in test_configs:
        print(f"\nLoading {name} from {ckpt_path}")
        model = load_model(ckpt_path, device, use_damping=use_damping)

        # loop over all requested tasks
        for current_task in args.task:
            key = (name, current_task)
            all_results[key] = {}
            print(f"\n{'='*60}")
            print(f"NAME: {name}, TASK: {current_task.upper()}")
            print(f"{'='*60}")

            ds, eval_fn = load_task(current_task)
        
            for num_loops in args.loops:
                acc = eval_fn(model, ds, num_loops, device, dtype, args.num_examples)
                all_results[key][num_loops] = acc
        del model
        torch.cuda.empty_cache()

    # print unified table
    print(f"\n{'='*80}")
    print("UNIFIED RESULTS TABLE")
    print(f"{'='*80}\n")

    # build header
    header = ['name', 'dataset'] + [str(l) for l in args.loops]
    print('\t'.join(header))

    # build rows
    csv_rows = [header]
    for name, _, _ in test_configs:
        for task in args.task:
            key = (name, task)
            if key in all_results:
                row = [name, task]
                for num_loops in args.loops:
                    acc = all_results[key].get(num_loops, None)
                    if acc is not None:
                        row.append(f"{acc:.4f}")
                    else:
                        row.append("N/A")
                print('\t'.join(row))
                csv_rows.append(row)

    # write to CSV with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f'eval_results_{timestamp}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"\nResults saved to {csv_path}")


if __name__ == '__main__':
    main()
