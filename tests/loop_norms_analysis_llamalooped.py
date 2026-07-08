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
"""

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

# ---- config ----
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
NUM_LOOPS = 4
# Modes to compare side by side on the same prompts.
RECUR_MODES = ["latent", "latent_renorm"]  # + "token", "soft"
SKIP_LAYERS = None  # e.g. {"1": [2]}; leave None so layers 0/-1 run each loop
N_SAMPLES = 10
MAX_NEW_TOKENS = 256
DEVICE = "cuda"
OUT_DIR = "eval/"

os.makedirs(OUT_DIR, exist_ok=True)


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
        ).to(DEVICE)
        prompt_len = enc["input_ids"].shape[1]

        # (1) instrumented prefill: capture loop input & end-of-loop hidden.
        captured_in.clear()
        captured_out.clear()
        recording["on"] = True
        with torch.no_grad():
            model(**enc)
        recording["on"] = False

        assert len(captured_in) == len(captured_out) == NUM_LOOPS, (
            f"expected {NUM_LOOPS} captures, got in={len(captured_in)} "
            f"out={len(captured_out)} (is layer 0 or -1 skipped in some loop?)"
        )
        in_norm = [c.float().norm(dim=-1).mean().item() for c in captured_in]
        out_norm = [c.float().norm(dim=-1).mean().item() for c in captured_out]

        # (2) generate the answer text (hooks disabled).
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
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

    def agg(key):
        return [
            sum(r[key][l] for r in records) / len(records)
            for l in range(NUM_LOOPS)
        ]

    return {
        "records": records,
        "agg_in": agg("loop_in_norm_mean"),
        "agg_out": agg("loop_out_norm_mean"),
    }


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = (
        LlamaLoopedForCausalLM.from_pretrained(
            MODEL,
            num_loops=NUM_LOOPS,
            recur_mode=RECUR_MODES[0],
            skip_layers=SKIP_LAYERS,
            dtype=torch.bfloat16,
        )
        .to(DEVICE)
        .eval()
    )

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

    questions = load_gsm8k_questions(N_SAMPLES)
    results = {}
    for mode in RECUR_MODES:
        print(f"\n### mode = {mode}")
        results[mode] = run_mode(
            model, tok, questions, mode, recording, captured_in, captured_out
        )

    pre_handle.remove()
    handle.remove()

    # ---- side-by-side comparison ----
    def print_table(title, key):
        print(f"\n=== {title} (mean over tokens & prompts) ===")
        header = "  loop  " + "".join(f"{m:>16}" for m in RECUR_MODES)
        print(header)
        for l in range(NUM_LOOPS):
            row = f"  {l:>4}  " + "".join(
                f"{results[m][key][l]:>16.3f}" for m in RECUR_MODES
            )
            print(row)

    print_table("per-loop input |h|", "agg_in")
    print_table("per-loop output |h|", "agg_out")

    if len(RECUR_MODES) >= 2:
        m0, m1 = RECUR_MODES[0], RECUR_MODES[1]
        same = sum(
            results[m0]["records"][i]["answer"] == results[m1]["records"][i]["answer"]
            for i in range(N_SAMPLES)
        )
        print(f"\n=== answers: {m0} vs {m1} identical: {same}/{N_SAMPLES} ===")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUT_DIR, f"loop_norms_cmp_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": MODEL,
                "num_loops": NUM_LOOPS,
                "recur_modes": RECUR_MODES,
                "skip_layers": SKIP_LAYERS,
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
