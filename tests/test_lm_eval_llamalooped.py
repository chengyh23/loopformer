import os
import json
from datetime import datetime
import lm_eval

output_path = "eval/"
os.makedirs(output_path, exist_ok=True)

# Evaluation backend:
#   "vllm": fast, use for full benchmarks. Injects loop config via hf_overrides.
#   "hf":   HuggingFace reference (slow, esp. on generative gsm8k). The HF
#           backend has no hf_overrides hook, so we bake the loop config into a
#           local checkpoint and register the Auto classes in-process.
backend = "vllm"

pretrained_model = "meta-llama/Llama-3.2-3B-Instruct"
max_model_len = 4096
# Where the HF looped checkpoint is materialized (backend="hf" only).
hf_ckpt_dir = "ckpts/llama_looped"
# Subsample for quick sanity runs (e.g. 50). None = full task.
limit = None

# Set to False to run the vanilla Llama baseline (stock LlamaForCausalLM, no
# looping) for comparison against the looped variant.
use_looped = True
num_loops = 4

# Recurrence mode for the looped variant:
#   "latent": hidden states flow directly into the next loop (continuous
#             residual). Default, backward-compatible.
#   "token":  between loops the hidden state is decoded to a token (greedy
#             argmax) and re-embedded (iterative self-conditioning). TP=1 only.
#   "soft":   next input = softmax(logits) @ E, the expected embedding under the
#             predicted distribution (soft bottleneck). Compile-friendly. TP=1.
recur_mode = "latent"

# skip_layers = {"1": [2], "3": [1]}
skip_layers = None

# tasks = ["hellaswag"]
tasks = ["gsm8k"]


def build_hf_ckpt_llama_looped(base, ckpt_dir):
    """Materialize a looped checkpoint from a stock Llama, baking in the loop
    config, so the HF backend can load it via AutoModelForCausalLM."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from models.llama_looped import LlamaLoopedConfig, LlamaLoopedForCausalLM

    # Register so AutoModelForCausalLM.from_pretrained resolves the looped class
    # (config.json model_type == "llama_looped"); no remote code needed.
    AutoConfig.register("llama_looped", LlamaLoopedConfig)
    AutoModelForCausalLM.register(LlamaLoopedConfig, LlamaLoopedForCausalLM)

    if not os.path.exists(os.path.join(ckpt_dir, "config.json")):
        model = LlamaLoopedForCausalLM.from_pretrained(
            base,
            num_loops=num_loops,
            recur_mode=recur_mode,
            skip_layers=skip_layers,
            dtype="bfloat16",
        )
        model.save_pretrained(ckpt_dir)
        AutoTokenizer.from_pretrained(base).save_pretrained(ckpt_dir)
    return ckpt_dir


if backend == "vllm":
    if use_looped:
        hf_overrides_dict = {
            # Load the standard Llama checkpoint as the looped (recurrent-depth)
            # variant by overriding the architecture.
            "architectures": ["LlamaLoopedForCausalLM"],
            "num_loops": num_loops,
            "recur_mode": recur_mode,
            "skip_layers": skip_layers,
        }
    else:
        # Baseline: stock Llama, no architecture override, no looping.
        hf_overrides_dict = {}

    # Pass model_args as a dict (not a string): lm-eval comma-splits string
    # model_args, which shatters nested JSON like hf_overrides. A dict is
    # forwarded via create_from_arg_obj() untouched, so hf_overrides stays a
    # real dict and reaches vLLM's LLM(hf_overrides=...) -> config.
    model = "vllm"
    model_args = {
        "pretrained": pretrained_model,
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.5,
        # Each loop allocates its own KV cache (num_loops * num_layers caches),
        # so keep max_model_len modest to avoid KV-cache OOM.
        "max_model_len": max_model_len,
        "hf_overrides": hf_overrides_dict,
    }
    # "token" recurrence does a data-dependent argmax + embedding gather inside
    # the forward, which can break torch.compile / CUDA-graph capture. Run eager.
    if use_looped and recur_mode == "token":
        model_args["enforce_eager"] = True
    batch_size = "auto"

elif backend == "hf":
    pretrained = build_hf_ckpt_llama_looped(pretrained_model, hf_ckpt_dir) if use_looped \
        else pretrained_model
    model = "hf"
    model_args = {
        "pretrained": pretrained,
        "trust_remote_code": True,
        "dtype": "bfloat16",
    }
    batch_size = 16  # HF benefits from a larger batch

else:
    raise ValueError(f"Unknown backend: {backend!r}")


results = lm_eval.simple_evaluate(
    model=model,
    model_args=model_args,
    tasks=tasks,
    batch_size=batch_size,
    limit=limit,
)


if results is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_path, f"results_{timestamp}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        # 使用 default=str 兜底处理 numpy 或其他特殊类型的评估指标对象
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)

    print(f"评估完成！结果已成功保存至: {file_path}")

    # 如果你想顺便在控制台打印一下命令行那种漂亮的表格，可以加这一行：
    if "results" in results:
        from lm_eval.utils import make_table
        print(make_table(results))
