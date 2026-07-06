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
