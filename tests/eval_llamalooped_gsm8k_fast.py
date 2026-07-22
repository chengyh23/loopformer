"""Fast GSM8K evaluation of LoopedLlama using direct transformers inference.

Unlike lm_eval (which is slow for long-generation tasks), this directly uses
transformers for faster evaluation.

Run (single GPU):
    CUDA_VISIBLE_DEVICES=1 python tests/eval_llamalooped_gsm8k_fast.py

Run with custom parameters:
    python tests/eval_llamalooped_gsm8k_fast.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --num_loops 4 \
        --recur_mode latent \
        --use_lora \
        --lora_adapter_dir ckpts/llama_looped_lora/adapter \
        --n_samples 10 \
        --max_new_tokens 256
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# Allow running as `python tests/eval_llamalooped_gsm8k_fast.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llama_looped import LlamaLoopedForCausalLM


def load_gsm8k_test_set(n=None):
    """Load GSM8K test set."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    if n:
        ds = ds.select(range(min(n, len(ds))))
    return ds


def extract_answer(text):
    """Extract final numeric answer from GSM8K response."""
    # Look for "#### " marker
    if "####" in text:
        return text.split("####")[-1].strip()
    return text.strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Fast GSM8K evaluation of LoopedLlama")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Model name or path")
    parser.add_argument("--num_loops", type=int, default=4,
                        help="Number of recurrent loops")
    parser.add_argument("--recur_mode", type=str, default="latent",
                        help="Recurrence mode (latent, etc.)")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Use LoRA adapter if available")
    parser.add_argument("--no_lora", dest="use_lora", action="store_false",
                        help="Disable LoRA adapter")
    parser.add_argument("--lora_adapter_dir", type=str, default="ckpts/llama_looped_lora/adapter",
                        help="Path to LoRA adapter directory")
    parser.add_argument("--use_loop_attn_residual", action="store_true",
                        help="Model was trained with Attention Residuals over the "
                        "loop-depth; loads loop_attn_res.pt from the adapter dir")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of samples to evaluate (None = full test set)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum new tokens to generate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda, cpu, etc.)")
    parser.add_argument("--out_dir", type=str, default="eval/",
                        help="Output directory for results")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[INFO] Loading model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)

    # model = AutoModelForCausalLM.from_pretrained(
    model = LlamaLoopedForCausalLM.from_pretrained(
        args.model,
        num_loops=args.num_loops,
        recur_mode=args.recur_mode,
        use_loop_attn_residual=args.use_loop_attn_residual,
        dtype=torch.bfloat16,
    )

    # If using LoRA, load and merge the adapter
    if args.use_lora and os.path.exists(args.lora_adapter_dir):
        print(f"[INFO] Loading and merging LoRA adapter...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_adapter_dir)
        model = model.merge_and_unload()

    # AttnRes weights are trained fully (not LoRA) and saved separately; load
    # them after the LoRA merge (they are untouched by merge_and_unload).
    if args.use_loop_attn_residual:
        attn_res_path = os.path.join(args.lora_adapter_dir, "loop_attn_res.pt")
        if os.path.exists(attn_res_path):
            print(f"[INFO] Loading loop-attn-res weights from {attn_res_path}")
            model.load_state_dict(torch.load(attn_res_path), strict=False)
        else:
            print(f"[WARN] --use_loop_attn_residual set but {attn_res_path} not found")

    model = model.to(args.device).eval()
    print(f"[INFO] Model ready on {args.device}\n")

    print(f"[INFO] Loading GSM8K test set (n={args.n_samples})...")
    dataset = load_gsm8k_test_set(args.n_samples)
    print(f"[INFO] Loaded {len(dataset)} samples\n")

    results = []
    correct = 0

    for idx, sample in enumerate(dataset):
        question = sample["question"]
        expected_answer = sample["answer"]  # Full answer with reasoning

        # Format as chat
        enc = tok.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(args.device)

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        prompt_len = enc["input_ids"].shape[1]
        generated_text = tok.decode(gen[0, prompt_len:], skip_special_tokens=True)
        generated_answer = extract_answer(generated_text)
        expected_final = extract_answer(expected_answer)

        # Simple check: both answers contain the same final number
        is_correct = generated_answer == expected_final

        if is_correct:
            correct += 1

        results.append({
            "idx": idx,
            "question": question,
            "generated": generated_text,
            "expected": expected_answer,
            "is_correct": is_correct,
        })

        progress = f"[{idx+1:4d}/{len(dataset)}]"
        status = "✓" if is_correct else "✗"
        print(f"{progress} {status} {question[:60]}...")

        # Print every 10 samples
        if (idx + 1) % 10 == 0:
            acc = correct / (idx + 1)
            print(f"  → Accuracy so far: {acc:.2%} ({correct}/{idx+1})\n")

    # Final stats
    accuracy = correct / len(dataset)
    print(f"\n{'='*70}")
    print(f"Final Accuracy: {accuracy:.2%} ({correct}/{len(dataset)})")
    print(f"{'='*70}\n")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lora_suffix = "_lora" if args.use_lora else ""
    out_path = os.path.join(args.out_dir, f"gsm8k_eval{lora_suffix}_{ts}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "num_loops": args.num_loops,
                "recur_mode": args.recur_mode,
                "use_lora": args.use_lora,
                "num_samples": len(dataset),
                "accuracy": accuracy,
                "correct": correct,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved results to: {out_path}\n")


if __name__ == "__main__":
    main()
