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

## Llama Looped Finetuning w/ Lora

meta-llama/Llama-3.2-3B-Instruct

|        Task          | GSM8K (flexible-extract) |
|----------------------|-------|
| llama                | 0.7718 |
| llama looped         | 0.0106 |
| llama looped w/ lora | 0.6535 |

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
