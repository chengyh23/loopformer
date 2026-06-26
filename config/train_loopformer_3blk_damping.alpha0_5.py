# identical to train_loopformer_3blk.py except:
#   - init_from / hf_model_name: start from open-sourced checkpoint
#   - use_damping: enable loop-level VP damping
#   - out_dir

batch_size = 4  # 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

max_iters = 2000
lr_decay_iters = 2000

out_dir = "runs/loopformer_damping-alpha0_5"
dataset = "openwebtext"

eval_interval = 200
eval_iters = 100
log_interval = 10

weight_decay = 2e-1

model_type = 'loopformer'
max_model_loops = 8
n_layer = 3
n_head = 32
n_embd = 2048

init_from = 'hf'
hf_model_name = 'armenjeddi/LoopFormer-3block-8iterations'
use_damping = True
loop_alpha = 0.5
device = 'cuda:0'