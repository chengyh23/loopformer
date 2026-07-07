CUDA_VISIBLE_DEVICES=6 lm-eval --model vllm \
  --model_args pretrained=ByteDance/Ouro-1.4B,trust_remote_code=True,gpu_memory_utilization=0.5,hf_overrides='{"skip_layers": {"0": [1, 4, 6}}}' \
  --tasks hellaswag \
  --batch_size auto \
  --output_path eval/