"""Compare recurrence modes of the looped Llama on GSM8K prompts: measure the
per-loop input/output hidden-state norms and the generated answers.

The first / last decoder layers each run once per loop, so a forward_pre_hook on
layers[0] captures each loop's INPUT hidden state and a forward_hook on
layers[-1] captures its OUTPUT (end-of-loop). We record those on the prompt's
prefill forward, then generate the answer separately.

Modes are switched at runtime on a single loaded model (recur_mode is read live
in forward), so weights are loaded only once.

Run (single GPU):
    CUDA_VISIBLE_DEVICES=6 python tests/loop_norms_analysis_llamalooped.py

Run with custom parameters:
    python tests/loop_norms_analysis_llamalooped.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --num_loops 4 \
        --recur_modes latent token soft \
        --n_samples 10
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
from transformers import AutoTokenizer

# Allow running as `python tests/loop_norms_analysis_llamalooped.py`: put the
# repo root (parent of tests/) on sys.path so `models` is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llama_looped import LlamaLoopedForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze loop norms in LoopedLlama")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Model name or path")
    parser.add_argument("--num_loops", type=int, default=4,
                        help="Number of recurrent loops")
    parser.add_argument("--recur_modes", type=str, nargs="+", default=["latent"],
                        help="Recurrence modes to compare (latent, token, soft, etc.)")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of GSM8K prompts to analyze")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum tokens to generate per answer")
    parser.add_argument("--use_lora", action="store_true", default=False,
                        help="Use LoRA adapter if available")
    parser.add_argument("--lora_adapter_dir", type=str, default="ckpts/llama_looped_lora/adapter",
                        help="Path to LoRA adapter directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda, cpu, etc.)")
    parser.add_argument("--output_dir", type=str, default="eval/",
                        help="Output directory for results")
    return parser.parse_args()


args = parse_args()
os.makedirs(args.output_dir, exist_ok=True)

print("\n" + "="*70)
print("[CONFIG] Loop Norms Analysis Settings:")
print("="*70)
for key, value in vars(args).items():
    print(f"  {key:30s} = {value}")
print("="*70 + "\n")


def load_gsm8k_questions(n):
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="test")
    return [ds[i]["question"] for i in range(n)]


def run_mode(model, tok, questions, mode, recording, captured_in, captured_out):
    """Run all prompts under `mode`; return per-prompt records + aggregates."""
    model.model.recur_mode = mode
    records = []
    for qi, q in enumerate(questions):
        # transformers 5.x apply_chat_template defaults to return_dict=True, so
        # this is a BatchEncoding; pass it with **enc, not positionally.
        enc = tok.apply_chat_template(
            [{"role": "user", "content": q}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(args.device)
        prompt_len = enc["input_ids"].shape[1]

        # (1) instrumented prefill: capture loop input & end-of-loop hidden.
        captured_in.clear()
        captured_out.clear()
        recording["on"] = True
        with torch.no_grad():
            model(**enc)
        recording["on"] = False

        assert len(captured_in) == len(captured_out) == args.num_loops, (
            f"expected {args.num_loops} captures, got in={len(captured_in)} "
            f"out={len(captured_out)} (is layer 0 or -1 skipped in some loop?)"
        )
        in_norm = [c.float().norm(dim=-1).mean().item() for c in captured_in]
        out_norm = [c.float().norm(dim=-1).mean().item() for c in captured_out]

        # (2) generate the answer text (hooks disabled).
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        answer = tok.decode(gen[0, prompt_len:], skip_special_tokens=True)

        records.append(
            {
                "idx": qi,
                "prompt_tokens": prompt_len,
                "loop_in_norm_mean": in_norm,
                "loop_out_norm_mean": out_norm,
                "answer": answer,
            }
        )
        in_str = ", ".join(f"{v:.1f}" for v in in_norm)
        out_str = ", ".join(f"{v:.1f}" for v in out_norm)
        print(f"  [{mode} {qi}] in:[{in_str}]  out:[{out_str}]")
        # print(f"  [{mode} {qi}] question: {q}")  # debug
        # print(f"  [{mode} {qi}] answer: {answer}")  # debug

    def agg(key):
        return [
            sum(r[key][l] for r in records) / len(records)
            for l in range(args.num_loops)
        ]

    return {
        "records": records,
        "agg_in": agg("loop_in_norm_mean"),
        "agg_out": agg("loop_out_norm_mean"),
    }


def main():
    print("[INFO] Starting loop norms analysis...\n")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = LlamaLoopedForCausalLM.from_pretrained(
        args.model,
        num_loops=args.num_loops,
        recur_mode=args.recur_modes[0],
        skip_layers=None,
        dtype=torch.bfloat16,
    )

    # If using LoRA, load and merge the adapter
    if args.use_lora and os.path.exists(args.lora_adapter_dir):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_adapter_dir)
        model = model.merge_and_unload()

    model = model.to(args.device).eval()

    recording = {"on": False}
    captured_in: list[torch.Tensor] = []
    captured_out: list[torch.Tensor] = []

    def pre_hook(_module, args):
        if recording["on"]:
            captured_in.append(args[0].detach())  # 1st arg = hidden_states

    def hook(_module, _inp, out):
        if recording["on"]:
            captured_out.append((out[0] if isinstance(out, tuple) else out).detach())

    pre_handle = model.model.layers[0].register_forward_pre_hook(pre_hook)
    handle = model.model.layers[-1].register_forward_hook(hook)

    questions = load_gsm8k_questions(args.n_samples)
    results = {}
    for mode in args.recur_modes:
        print(f"\n### mode = {mode}")
        results[mode] = run_mode(
            model, tok, questions, mode, recording, captured_in, captured_out
        )

    pre_handle.remove()
    handle.remove()

    # ---- side-by-side comparison ----
    def print_table(title, key):
        print(f"\n=== {title} (mean over tokens & prompts) ===")
        header = "  loop  " + "".join(f"{m:>16}" for m in args.recur_modes)
        print(header)
        for l in range(args.num_loops):
            row = f"  {l:>4}  " + "".join(
                f"{results[m][key][l]:>16.3f}" for m in args.recur_modes
            )
            print(row)

    print_table("per-loop input |h|", "agg_in")
    print_table("per-loop output |h|", "agg_out")

    if len(args.recur_modes) >= 2:
        m0, m1 = args.recur_modes[0], args.recur_modes[1]
        same = sum(
            results[m0]["records"][i]["answer"] == results[m1]["records"][i]["answer"]
            for i in range(args.n_samples)
        )
        print(f"\n=== answers: {m0} vs {m1} identical: {same}/{args.n_samples} ===")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lora_suffix = "_lora" if args.use_lora else ""
    out_path = os.path.join(args.output_dir, f"loop_norms_cmp{lora_suffix}_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "num_loops": args.num_loops,
                "recur_modes": args.recur_modes,
                "skip_layers": None,
                "use_lora": args.use_lora,
                "questions": questions,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
