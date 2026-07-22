# LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation (ICLR 2026)

<a target="_blank" href="">
  <img style="height:22pt" src="https://img.shields.io/badge/-Paper-red?style=flat&logo=arxiv">
</a>
<a target="_blank" href="https://loopformer.github.io/">
  <img style="height:22pt" src="https://img.shields.io/badge/-🌐%20Website-blue?style=flat">
</a>
<a target="_blank" href="https://huggingface.co/collections/armenjeddi/loopformer">
  <img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat">
</a>

**Authors:**  
[Ahmadreza Jeddi](https://armenjeddi.github.io/), [Marco Ciccone](https://marcociccone.github.io/), [Babak Taati](https://www.cs.toronto.edu/~taati/)
<br>

![LoopFormer](assets/loopformer.png)

---

This repository contains the official implementation of **LoopFormer**.

The codebase is a fork of **NanoGPT**, and we intentionally keep it as close as possible to the original implementation for clarity and reproducibility. Beyond the looped / elastic-depth components, the main architectural difference is using **RMSNorm** instead of **LayerNorm**.

---

## Installation

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```
## Training

```bash
python train.py config/train_loopformer_3blk_damping.py
python train.py config/train_loopformer_3blk_wo_damping.py
```

## Evaluation
```bash
python eval_poc.py --task winogrande  --loops 4 8 12 16 20 24    --num_examples 500
```
## Attention Frobenius Norm Analysis

To understand how attention patterns evolve across loop iterations and layers, we analyze the Frobenius norm between causal self-attention matrices at every pair of `(loop_idx, layer_idx)` depths. Under `mechanistic_analysis/`

### Scripts

- `attn_frobenius_gsm8k.py` — Compute and plot F-norm heatmaps on GSM8K (reasoning task)
- `attn_frobenius_hellaswag.py` — Compute and plot F-norm heatmaps on HellaSwag (non-reasoning task)

Usage:
```bash
python attn_frobenius_gsm8k.py --num_loops 8 --num_examples 20
python attn_frobenius_hellaswag.py --num_loops 24 --num_examples 20
```

### Key Findings

**Reasoning tasks (GSM8K)**: Attention patterns **evolve substantially across all loop iterations**. The Frobenius norm between distant loops remains large (deep blue in heatmaps), indicating the model continuously refines its attention focus even after many iterations. This suggests that complex reasoning requires multiple refinement passes over the hidden state.

**Non-reasoning tasks (HellaSwag)**: Attention patterns **stabilize quickly within the first loop**. Frobenius norms are ~5× smaller than GSM8K, and remain uniformly low across all loop pairs, indicating that simple commonsense completion requires minimal iterative refinement.

**Extrapolation behavior**: When pushed beyond the training distribution (8 iterations), reasoning tasks show continued evolution, while non-reasoning tasks exhibit flat patterns, suggesting different generalization characteristics.

## Layer Skip Analysis

To study the role of individual layers in Ouro, we implement selective layer skipping during inference.

Files

- `tests/gen_ouro_random_skip.py` — Custom generation 
- `tests/test_lm_eval.py` — Evaluate on benchmark


### Results

**Configuration (Ouro)** ByteDance/Ouro-1.4B

```python
"skip_layers": {"2": [6, 10, 15, 17, 19, 20]}
"skip_layers": {"2": [15, 16, 17, 18, 19, 20]}
"skip_layers": {"2": [6, 7, 8, 18, 19, 20]}
"skip_layers": {"2": [6, 7, 14, 15, 19, 20]}
"skip_layers": {"1": [14, 15], "2": [6, 7], "3": [19, 20]}
"skip_layers": {"1": [6, 7], "2": [14, 15], "3": [19, 20]}
"skip_layers": {"1": [6, 7], "2": [6, 7], "3": [6, 7]}
"skip_layers": {"1": [19, 20], "2": [19, 20], "3": [19, 20]}
```

| Task | Full | Skip |skip_layers|
|------|------|------|------|
| HellaSwag | 0.7161 | 0.6552 | {"2": [6, 10, 15, 17, 19, 20]}  |
| GSM8K | 0.7938 | 0.3973 | {"2": [6, 10, 15, 17, 19, 20]}  |
| GSM8K |        | 0.5072 | {"2": [15, 16, 17, 18, 19, 20]} |
| GSM8K |        | 0.5246 | {"2": [6, 7, 8, 18, 19, 20]}  |
| GSM8K |        | 0.4928 | {"2": [6, 7, 14, 15, 19, 20]} |
| GSM8K |        | 0.3389 | {"1": [14, 15], "2": [6, 7], "3": [19, 20]} |
| GSM8K |        | 0.3897 | {"1": [6, 7], "2": [14, 15], "3": [19, 20]} |
| GSM8K |        | 0.2290 | {"1": [6, 7], "2": [6, 7], "3": [6, 7]} |
| GSM8K |        | 0.4329 | {"1": [19, 20], "2": [19, 20], "3": [19, 20]} |



gsm8k metric: flexible-extract

**Configuration (LoopFormer)** armenjeddi/LoopFormer-3block-8iterations-FineWeb300K
```python
"skip_layers": {"1": [2], "3": [1], "6": [0]}
```

| Task | Full | Skip |
|------|------|------|
| HellaSwag | 0.4181 | 0.3922 |

**Configuration (Llama)** meta-llama/Llama-3.2-3B-Instruct

`skip_layers = {"1": [2], "3": [1]}`

|    Task     | HellaSwag | GSM8K |
|-------------|------|------|
| Llama       | 0.7167 | 0.6793 |
| LlamaLooped | 0.4569 | 0.0083 |
| LlamaLooped w/ skip | 0.4631 |  |
| LlamaLooped recur (token) |  | 0.0159 |
| LlamaLooped recur (soft) |  | 0.0068 |
| LlamaLooped (2 loops) |  | 0.0144 |
## Llama Looped w/ sandwich norm
Run `python train/train_lora_llamalooped.py --model meta-llama/Llama-3.2-1B-Instruct --sandwich-norm`

## Llama Looped w/ AttnRes
`LoopAttnResidual`: aggregates prior hidden states.

## Llama Looped Finetuning w/ Lora

Train `CUDA_VISIBLE_DEVICES=6 python train/train_lora_llamalooped.py --model meta-llama/Llama-3.2-1B-Instruct`
Enable per-loop adapter with `--per-loop-lora`

Eval `CUDA_VISIBLE_DEVICES=5 python tests/test_lm_eval_llamalooped.py --use_looped --use_lora --model meta-llama/Llama-3.2-1B-Instruct --lora_adapter_dir ckpts/Llama-3.2-1B-Instruct_looped_lora/adapter --gpu_memory_utilization 0.3`

meta-llama/Llama-3.2-3B-Instruct (llama-3b)


|        Task                  | GSM8K (flexible-extract) | MBPP (pass_at_1) |
|------------------------------|--------------------------|--------------|
| llama-3b                                                | 0.7718 | |
| llama-3b looped                                         | 0.0106 | |
| llama-3b looped w/ lora                                 | 0.6535 | |
| llama-3b looped w/ per-loop lora                        |  | |
| llama-1b                                                | 0.4246 | 0.364 |
| llama-1b looped                                         | 0.0136 | |
| llama-1b looped w/ lora                                 | 0.3465 | |
| llama-1b looped w/ per-loop lora                        | 0.3730 | |
| llama-1b looped w/ lora w/ CoT-supervision (50k)        | 0.2191 | |
| llama-1b looped w/ lora w/ CoT-supervision              | 0.4329 | 0 |
| llama-1b looped w/ lora w/ CoT-supervision (num_loops=3) | 0.4503 | |
| llama-1b looped w/ lora w/ CoT-supervision (num_loops=2) | 0.4208 | |
| llama-1b looped w/ lora w/ CoT-supervision (num_loops=1) | 0.3973 | |
| llama-1b looped w/ lora w/ CoT-supervision (num_loops=5) | 0.4102 | |
| llama-1b looped w/ lora w/ CoT-supervision (num_loops=6) | 0.3738 | |
| llama-1b w/ lora (gsm8k-aug SFT, no loop) | **0.4845** | 0.354 |
| llama-1b looped w/ lora w/ CoT-supervision w/fsw=0.3 | 0.4405 | 0 |
| llama-1b looped w/ lora w/ CoT-supervision w/fsw=0.0 | 0.0455 | |
| llama-1b looped prelude2-coda2 w/ lora                  | 0.3146 | |
| llama-1b looped prelude4-coda4 w/ lora                  | 0.3146 | |


**findings**
- **Naive looping is catastrophic; the failure is low-rank fixable.** Re-entering the decoder stack collapses accuracy (0.42 → 0.01) — the re-entry hidden states are far outside the input distribution each layer was trained on — yet a <1%-parameter LoRA recovers most of it, so the drift is largely correctable by a low-rank adaptation.
- **The controlled comparison shows the gain comes from data, not from looping.** The same LoRA recipe on the same GSM8K-Aug data *without* looping reaches 0.4845 — above every looped variant (best: 0.4503). At matched data, 4× serial depth currently buys negative return.


### Loop-Aligned CoT stepwise Supervision

**Aligning loop depth with reasoning depth**: each loop iteration is supervised to decode one CoT step.
Dataset: [GSM8K-Aug](https://huggingface.co/datasets/whynlp/gsm8k-aug), 


Run `CUDA_VISIBLE_DEVICES=6 python train/train_lora_llamalooped_cot.py --model meta-llama/Llama-3.2-1B-Instruct --n-train 50000 --batch-size 64 --grad-accum 4 [--per-loop-lora] [--final-step-weight FINAL_STEP_WEIGHT]`

**Downweighting the final loop's step-region loss**

Enable downweighting final loop's step-region tokens' CE (weighted mean: step tokens × `w`, answer tokens × 1.0) with `--final-step-weight 0.3`

`w = 1.0` (default) is exactly the unweighted loss. the last loop is trained on the full response (all steps + answer), which duplicates the intermediate loops' step supervision and lets step tokens dominate over answer tokens — the last loop degenerates into ordinary SFT and dilutes the loop division of labor. 

###  Latent Carry Loop-Aligned CoT stepwise Supervision

Run `CUDA_VISIBLE_DEVICES=6 python train/train_lora_llamalooped_latent.py --model meta-llama/Llama-3.2-1B-Instruct --per-loop-lora --batch-size 32 --grad-accum 8`

Eval `tests/eval_latent_gsm8k.py` with adapter in `ckpts/Llama-3.2-1B-Instruct_looped_lora_latent_perloop`

### Prelude-Recursive core-Coda 
Run `CUDA_VISIBLE_DEVICES=6 python train/train_lora_llamalooped.py --model meta-llama/Llama-3.2-1B-Instruct --prc 2 2`

Eval `python tests/test_lm_eval_llamalooped.py --use_looped --use_lora --model meta-llama/Llama-3.2-1B-Instruct --lora_adapter_dir ckpts/Llama-3.2-1B-Instruct_looped_lora_cot_prc2-2/adapter --prc 2 2 --gpu_memory_utilization 0.3`
## Loop Norm Analysis

num_loops=4

|        Task          | norm (mean) |
|----------------------|-------|
| llama looped         | [1.019, 43.573, 73.341, 121.094, 190.850] |
| llama looped w/ lora | [1.019, 66.619, 149.975, 199.049, 234.016] |


## Citation
If you find this work useful, please give us a citation:
```bibtex
@misc{jeddi2026loopformerelasticdepthloopedtransformers,
      title={LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation}, 
      author={Ahmadreza Jeddi and Marco Ciccone and Babak Taati},
      year={2026},
      eprint={2602.11451},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.11451}, 
}
```
