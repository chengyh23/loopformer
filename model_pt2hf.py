import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# model_dir = "loopformer_damping-alpha0_5"
model_dir = "loopformer_damping"


root_path_in = "/home/yc714/proj/decmas/loopformer/runs"
root_path_out = "/home/yc714/proj/decmas/loopformer/tmp"
ckpt = torch.load(
    os.path.join(root_path_in, model_dir, "ckpt.pt"), 
    map_location="cpu",
    weights_only=False,
)

state_dict = ckpt["model"]
new_state_dict = {}
for k, v in state_dict.items():
    # HF 没有这个参数
    if k == "loop_alpha":
        continue
    new_state_dict["gpt." + k] = v


model = AutoModelForCausalLM.from_pretrained(
    "armenjeddi/LoopFormer-3block-8iterations",
    trust_remote_code=True,

)
model.load_state_dict(new_state_dict)

model.save_pretrained(
    os.path.join(root_path_out, model_dir),
    safe_serialization=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    "armenjeddi/LoopFormer-3block-8iterations",
    trust_remote_code=True,
)
tokenizer.save_pretrained(os.path.join(root_path_out, model_dir))