"""Generate Ouro responses under randomly generated per-ut_step layer skips.

Uses the vLLM engine directly (LLM/SamplingParams). For each random skip spec
we call `set_skip_layers` on the model inside the engine worker, then generate.

Run:
    conda run -n loopedlm python loopformer/tests/gen_ouro_random_skip.py
    # options: --model PATH --seed 0 --num-configs 3 --max-skip-frac 0.5
"""

import argparse
import os
import random
from pathlib import Path

# `apply_model` ships the callable to the worker process; allow cloudpickle so
# the closure below can be serialized. Must be set before importing vllm.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

from vllm import LLM, SamplingParams  # noqa: E402

# DEFAULT_MODEL = str(Path(__file__).resolve().parent.parent / "Ouro-1.4B")
# DEFAULT_MODEL = "ByteDance/Ouro-1.4B"
DEFAULT_MODEL = "armenjeddi/LoopFormer-3block-8iterations-FineWeb300K"

PROMPTS = [
    "The capital of France is",
    "Once upon a time,",
    "Q: What is 2 + 2? A:",
]

def get_skip_layer_config():
    """
    Returns a list of (name, skip_spec) tuples
    """
    configs = []
    # configs.append(("baseline", {}))
    # for i in range(args.num_configs):
    #     configs.append(
    #         (f"random#{i}", random_skip_spec(
    #             total_ut_steps, num_layers, rng, args.max_skip_frac))
    #     )
    # _config = {   # Ruins generation
    #     0: [1, 8, 12, 13, 15, 16], 
    #     1: [1, 4, 6, 9, 11, 12, 15, 16, 17, 18, 20, 23], 
    #     2: [1, 2, 3, 4, 5, 8, 9, 10, 15, 17, 19, 23], 
    #     3: [6, 10, 15, 17, 19, 20]
    # }
    # configs.append(("random#0", _config))
    
    # # In each loop, skip 6/24 layers
    # # Skipping one of the loops does not ruin generation. 
    # # Skipping all loops also generates reasonable generation.
    # _config = {
    #     0: [6, 10, 15, 17, 19, 20],
    #     1: [6, 10, 15, 17, 19, 20],
    #     2: [6, 10, 15, 17, 19, 20],
    #     3: [6, 10, 15, 17, 19, 20]
    # }
    # configs.append(("manual#0", _config))

    # # In each loop, skip 12/24 layers
    # # Skipping one of the loops does not ruin generation. 
    # # Skipping two of the loops partially ruins generation. 
    # # Skipping all loops ruins generation.
    # _config = {
    #     # 0: [1, 4, 6, 9, 11, 12, 15, 16, 17, 18, 20, 23], 
    #     1: [1, 4, 6, 9, 11, 12, 15, 16, 17, 18, 20, 23], 
    #     2: [1, 4, 6, 9, 11, 12, 15, 16, 17, 18, 20, 23], 
    #     # 3: [1, 4, 6, 9, 11, 12, 15, 16, 17, 18, 20, 23], 
    # }
    # configs.append(("manual#1", _config))

    # In each loop, skip 6/24 layers. 
    # ruins generation   
    _config = {
        # 0: [1, 8, 12, 13, 14, 16], 
        1: [3, 7, 9, 18, 21, 22],
        # 1: [6, 10, 15, 17, 19, 20],
        # 2: [0, 2, 4, 5, 11, 23],
        3: [6, 10, 15, 17, 19, 20]
    }
    configs.append(("manual#2", _config))

    return configs

def get_skip_layer_config_loopformer():
    configs = []
    _config = {
        1: [2],
        3: [1],
        6: [0],
    }
    configs.append(("manual#0", _config))

    return configs


def random_skip_spec(total_ut_steps, num_layers, rng, max_skip_frac):
    """Random {ut_step: [layer_idx, ...]} skipping up to max_skip_frac layers."""
    max_skip = max(0, int(num_layers * max_skip_frac))
    spec = {}
    for ut_step in range(total_ut_steps):
        k = rng.randint(0, max_skip)
        if k:
            spec[ut_step] = sorted(rng.sample(range(num_layers), k))
    return spec

def main2():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-configs", type=int, default=1)
    parser.add_argument("--max-skip-frac", type=float, default=0.5)
    args = parser.parse_args()

    configs = get_skip_layer_config()
    skip_layers = {
        "0": [1, 4, 6, 9, 11, 12, 15, 16, 17, 18, 20, 23], 
        "1": [1, 4, 6, 9, 11, 12, 15, 16, 17, 18, 20, 23], 
        "3": [1, 4, 6, 9, 11, 12, 15, 16, 17, 18, 20, 23]
    }
    # 1. Init the model on the vLLM engine.
    #    enforce_eager=True avoids torch.compile recompiles when we change the
    #    skip config between generate() calls.
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        max_model_len=2048,
        hf_overrides={"skip_layers": skip_layers},
    )

    # 2. Sampling params (greedy so skip effects are clearly visible).
    sampling_params = SamplingParams(temperature=0.8, max_tokens=64)

    # # Read the recurrence shape from the loaded model.
    # total_ut_steps, num_layers = llm.apply_model(
    #     lambda model: (model.model.total_ut_steps, model.config.num_hidden_layers)
    # )[0]
    # print(f"\nLoaded Ouro: total_ut_steps={total_ut_steps}, "
    #       f"num_hidden_layers={num_layers}")

    # rng = random.Random(args.seed)

    # # 3. Baseline (no skips) followed by N random skip configs.
    # configs = get_skip_layer_config()
    # for name, spec in configs:
    #     # Apply the skip spec inside the engine worker.
    #     llm.apply_model(lambda model: model.set_skip_layers(spec))

    #     print("\n" + "=" * 70)
    #     print(f"[{name}] skip spec: {spec}")
    #     print("=" * 70)

    #     outputs = llm.generate(PROMPTS, sampling_params)
    #     for output in outputs:
    #         text = output.outputs[0].text
    #         print("-" * 50)
    #         print(f"[prompt] {output.prompt}")
    #         print(f"[generated]\n{text}")

    outputs = llm.generate(PROMPTS, sampling_params)
    for output in outputs:
        text = output.outputs[0].text
        print("-" * 50)
        print(f"[prompt] {output.prompt}")
        print(f"[generated]\n{text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-configs", type=int, default=1)
    parser.add_argument("--max-skip-frac", type=float, default=0.5)
    args = parser.parse_args()

    # 1. Init the Ouro model on the vLLM engine.
    #    enforce_eager=True avoids torch.compile recompiles when we change the
    #    skip config between generate() calls.
    if "Ouro" in args.model: max_model_len = 2048
    elif "LoopFormer" in args.model: max_model_len = None
    else: raise ValueError(f"Unknown model: {args.model}")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        max_model_len=max_model_len,
    )

    # 2. Sampling params (greedy so skip effects are clearly visible).
    sampling_params = SamplingParams(temperature=0.8, max_tokens=64)

    # Read the recurrence shape from the loaded model.
    def get_model_info(model):
        if "Ouro" in args.model:
            return (model.model.total_ut_steps, model.config.num_hidden_layers)
        elif "LoopFormer" in args.model:
            # LoopFormer is a GPT-based model with num_ut_steps (loop iterations)
            # and n_layer (depth per iteration)
            return (len(model.config.steps), model.config.n_layer)
        else:
            raise ValueError(f"Unknown model: {args.model}")

    # total_ut_steps, num_layers = llm.apply_model(get_model_info)[0]
    # print(f"\nLoaded model: total_ut_steps={total_ut_steps}, "
    #       f"num_layers={num_layers}")

    rng = random.Random(args.seed)

    # 3. Baseline (no skips) followed by N random skip configs.
    if "Ouro" in args.model:
        configs = get_skip_layer_config()
    elif "LoopFormer" in args.model:
        configs = get_skip_layer_config_loopformer()
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    for name, spec in configs:
        # Apply the skip spec inside the engine worker.
        llm.apply_model(lambda model: model.set_skip_layers(spec))

        print("\n" + "=" * 70)
        print(f"[{name}] skip spec: {spec}")
        print("=" * 70)

        outputs = llm.generate(PROMPTS, sampling_params)
        for output in outputs:
            text = output.outputs[0].text
            print("-" * 50)
            print(f"[prompt] {output.prompt}")
            print(f"[generated]\n{text}")


if __name__ == "__main__":
    main()
    # main2()