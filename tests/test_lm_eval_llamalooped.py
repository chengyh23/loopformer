import argparse
import os
import sys
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Set logging to WARNING to reduce overhead
logging.basicConfig(level=logging.WARNING)

# Make `models` importable when run as `python tests/test_lm_eval_llamalooped.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.llama_looped import LlamaLoopedForCausalLM

import lm_eval
from lm_eval.models.huggingface import HFLM


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LoopedLlama with lm_eval")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "hf"],
                        help="Evaluation backend: vllm (fast) or hf (reference)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Pretrained model name or path")
    parser.add_argument("--max_model_len", type=int, default=2048,
                        help="Maximum model sequence length (vllm only)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of samples to evaluate (None=full test set)")
    parser.add_argument("--use_looped", action="store_true", default=True,
                        help="Use looped variant")
    parser.add_argument("--no_looped", dest="use_looped", action="store_false",
                        help="Use vanilla baseline (no looping)")
    parser.add_argument("--num_loops", type=int, default=4,
                        help="Number of recurrent loops")
    parser.add_argument("--recur_mode", type=str, default="latent",
                        choices=["latent", "token", "soft"],
                        help="Recurrence mode: latent (residual), token (greedy argmax), or soft (softmax bottleneck)")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Use LoRA adapter if available")
    parser.add_argument("--no_lora", dest="use_lora", action="store_false",
                        help="Disable LoRA adapter")
    parser.add_argument("--lora_adapter_dir", type=str,
                        help="Path to LoRA adapter directory, such as ckpts/llama_looped_lora/adapter")
    parser.add_argument("--tasks", type=str, nargs="+", default=["gsm8k"],
                        help="Tasks to evaluate (gsm8k, hellaswag, etc.)")
    parser.add_argument("--output_dir", type=str, default="eval/",
                        help="Output directory for results")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5,
                        help="GPU memory utilization ratio (vllm only)")
    return parser.parse_args()


args = parse_args()
os.makedirs(args.output_dir, exist_ok=True)

assert args.use_looped or not args.use_lora, \
    "LoRA adapters are only for looped variant. Use --no_looped without LoRA or use --use_looped with LoRA."

print("\n" + "="*70)
print("[CONFIG] Evaluation Settings:")
print("="*70)
for key, value in vars(args).items():
    print(f"  {key:30s} = {value}")
print("="*70 + "\n")

if args.backend == "vllm":
    if args.use_looped:
        hf_overrides_dict = {
            "architectures": ["LlamaLoopedForCausalLM"],
            "num_loops": args.num_loops,
            "recur_mode": args.recur_mode,
            "skip_layers": None,
        }
    else:
        hf_overrides_dict = {}

    model = "vllm"
    model_args = {
        "pretrained": args.model,
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "hf_overrides": hf_overrides_dict,
        "lora_local_path": args.lora_adapter_dir if args.use_lora else None,
    }
    if args.use_looped and args.recur_mode == "token":
        model_args["enforce_eager"] = True
    batch_size = "auto"

elif args.backend == "hf":
    tok = AutoTokenizer.from_pretrained(args.model)

    if args.use_looped:
        pretrained = LlamaLoopedForCausalLM.from_pretrained(
            args.model,
            num_loops=args.num_loops,
            recur_mode=args.recur_mode,
            dtype="bfloat16",
        ).to("cuda")

        if args.use_lora:
            assert args.lora_adapter_dir, "--use_lora requires --lora_adapter_dir"
            from peft import PeftModel

            loop_names = [f"loop{i}" for i in range(args.num_loops)]
            is_per_loop = all(
                os.path.isdir(os.path.join(args.lora_adapter_dir, n))
                for n in loop_names
            )
            if is_per_loop:
                # One LoRA adapter per loop (dir holds loop0/, loop1/, ...).
                # set_loop_adapters activates adapter loop{i} while loop i runs,
                # via a per-decoder-layer forward pre-hook.
                pretrained = PeftModel.from_pretrained(
                    pretrained,
                    os.path.join(args.lora_adapter_dir, loop_names[0]),
                    adapter_name=loop_names[0],
                )
                for name in loop_names[1:]:
                    pretrained.load_adapter(
                        os.path.join(args.lora_adapter_dir, name), adapter_name=name
                    )
                pretrained.get_base_model().set_loop_adapters(loop_names)
                print(f"[INFO] Loaded {args.num_loops} per-loop LoRA adapters: {loop_names}")
            else:
                # Single adapter shared across all loops.
                pretrained = PeftModel.from_pretrained(pretrained, args.lora_adapter_dir)
                print(f"[INFO] Loaded shared LoRA adapter from {args.lora_adapter_dir}")
            pretrained = pretrained.to("cuda")
    else:
        pretrained = args.model

    model = "hf"
    model_args = {
        "pretrained": pretrained,
        "tokenizer": tok,
        "dtype": "bfloat16",
    }
    # batch_size = 1
    batch_size = "auto"

else:
    raise ValueError(f"Unknown backend: {args.backend!r}")


print("[INFO] Starting evaluation...\n")

import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"  # tasks: mbpp
results = lm_eval.simple_evaluate(
    model=model,
    model_args=model_args,
    # model=lm,
    tasks=args.tasks,
    confirm_run_unsafe_code=True,   # tasks: mbpp
    limit=args.limit,
    batch_size=batch_size,
    # max_batch_size=batch_size,
    # num_fewshot=0,
    # log_samples=False,  # Disable to reduce I/O
    # cache_requests=True,  # Cache requests
    # verbosity="INFO",
    apply_chat_template=True,  # Apply chat template for instruction-tuned models
    # gen_kwargs={"max_gen_toks": 256},  # Limit generation length
)


if results is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(args.output_dir, f"results_{timestamp}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        # 使用 default=str 兜底处理 numpy 或其他特殊类型的评估指标对象
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)

    print(f"评估完成！结果已成功保存至: {file_path}")

    # 如果你想顺便在控制台打印一下命令行那种漂亮的表格，可以加这一行：
    if "results" in results:
        from lm_eval.utils import make_table
        print(make_table(results))
