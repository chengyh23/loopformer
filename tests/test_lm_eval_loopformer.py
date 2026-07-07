import os
import json
from datetime import datetime
import lm_eval

output_path = "eval/"
os.makedirs(output_path, exist_ok=True)

pretrained_model = "armenjeddi/LoopFormer-3block-8iterations-FineWeb300K"
max_model_len = None
hf_overrides_dict = {
    "skip_layers": {
        "1": [2],
        "3": [1],
        "6": [0],
    }
}
# hf_overrides_dict = {}

tasks = ["hellaswag"]
# tasks = ["gsm8k"]

# Pass model_args as a dict (not a string): lm-eval comma-splits string
# model_args, which shatters nested JSON like hf_overrides. A dict is forwarded
# via create_from_arg_obj() untouched, so hf_overrides stays a real dict and
# reaches vLLM's LLM(hf_overrides=...) -> config.skip_layers.
model_args = {
    "pretrained": pretrained_model,
    "trust_remote_code": True,
    "gpu_memory_utilization": 0.3,
    "max_model_len": max_model_len,  # Do we need to set this param??
    "hf_overrides": hf_overrides_dict,
}

results = lm_eval.simple_evaluate(
    model="vllm",
    model_args=model_args,
    tasks=tasks,
    batch_size="auto",
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