"""
Plot the Frobenius norm between causal self-attention patterns at every pair
of (loop, layer) depths in a LoopFormer model on HellaSwag (non-reasoning task),
averaged over a batch of examples and over attention heads.

Usage:
    python attn_frobenius_hellaswag.py \
        [--ckpt armenjeddi/LoopFormer-3block-8iterations-FineWeb300K] \
        [--num_loops 8] [--num_examples 20] [--device cuda]
"""
import argparse
import math
import textwrap
import types
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
from datasets import load_dataset
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from eval_poc import load_model


def build_prompt(enc, ctx: str, ending: str, block_size: int) -> torch.Tensor:
    text = f"{ctx} {ending}"
    ids = enc.encode(text)
    return ids[-block_size:]


# ---------------------------------------------------------------------------
# attention capture: monkey-patch each CausalSelfAttention module
# ---------------------------------------------------------------------------

def install_attention_capture(model):
    store = {}
    call_counts = [0] * model.config.n_layer

    def make_forward(layer_idx):
        def forward(self, x):
            B, T, C = x.size()
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            att = att.masked_fill(~causal, float('-inf'))
            att = F.softmax(att, dim=-1)

            loop_idx = call_counts[layer_idx]
            call_counts[layer_idx] += 1
            store[(loop_idx, layer_idx)] = att.detach().float().cpu()

            att = self.attn_dropout(att)
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self.resid_dropout(self.c_proj(y))
            return y
        return forward

    for layer_idx, block in enumerate(model.transformer.h.blocks):
        block.attn.forward = types.MethodType(make_forward(layer_idx), block.attn)

    def reset():
        store.clear()
        for i in range(len(call_counts)):
            call_counts[i] = 0

    return store, reset


@torch.no_grad()
def depth_frobenius_matrix(model, ids: list, steps: list, device: str, dtype: torch.dtype,
                            store: dict, reset) -> np.ndarray:
    reset()
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.amp.autocast(device_type=device.split(':')[0], dtype=dtype):
        model(x, steps=steps)

    n_layer = model.config.n_layer
    num_loops = len(steps)
    D = num_loops * n_layer
    att_by_depth = [store[(loop_idx, layer_idx)]
                    for loop_idx in range(num_loops) for layer_idx in range(n_layer)]

    mat = np.zeros((D, D), dtype=np.float64)
    for d1 in range(D):
        for d2 in range(d1 + 1, D):
            diff = att_by_depth[d1] - att_by_depth[d2]
            frob = torch.linalg.matrix_norm(diff, ord='fro', dim=(-2, -1))  # (1, n_head)
            val = frob.mean().item()  # avg over batch & heads
            mat[d1, d2] = val
            mat[d2, d1] = val
    return mat


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"


def plot_depth_matrix(mat: np.ndarray, n_layer: int, num_loops: int, title: str, out_path: str):
    D = mat.shape[0]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)

    fig_w = max(7.5, D * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, max(5, D * 0.32)), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    im = ax.imshow(mat, cmap=cmap, vmin=0.0)

    labels = [f"{loop}-{layer}" for loop in range(num_loops) for layer in range(n_layer)]
    ax.set_xticks(range(D))
    ax.set_yticks(range(D))
    ax.set_xticklabels(labels, rotation=90, fontsize=6, color=MUTED_INK)
    ax.set_yticklabels(labels, fontsize=6, color=MUTED_INK)
    ax.tick_params(length=0)

    # loop-boundary gridlines
    for boundary in range(n_layer, D, n_layer):
        ax.axhline(boundary - 0.5, color=GRIDLINE, linewidth=1.0)
        ax.axvline(boundary - 0.5, color=GRIDLINE, linewidth=1.0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Frobenius norm (avg over batch & heads)", color=PRIMARY_INK, fontsize=9)
    cbar.ax.tick_params(labelsize=7, colors=MUTED_INK)

    ax.set_xlabel("depth (loop-layer)", color=PRIMARY_INK, fontsize=9)
    ax.set_ylabel("depth (loop-layer)", color=PRIMARY_INK, fontsize=9)
    wrap_width = max(40, int(fig_w * 8))
    wrapped_title = "\n".join(textwrap.wrap(title, width=wrap_width))
    ax.set_title(wrapped_title, color=PRIMARY_INK, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default="armenjeddi/LoopFormer-3block-8iterations-FineWeb300K")
    parser.add_argument('--use_damping', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_loops', type=int, default=8)
    parser.add_argument('--num_examples', type=int, default=20)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    print(f"Loading {args.ckpt}")
    model = load_model(args.ckpt, args.device, use_damping=args.use_damping)
    store, reset = install_attention_capture(model)

    enc = tiktoken.get_encoding("gpt2")
    print("Loading HellaSwag validation split...")
    ds = load_dataset("Rowan/hellaswag", split="validation")

    steps = [1 / args.num_loops] * args.num_loops
    D = args.num_loops * model.config.n_layer
    mat_sum = np.zeros((D, D), dtype=np.float64)

    n = min(args.num_examples, len(ds))
    for ex in tqdm(ds.select(range(n)), desc="examples"):
        # use context + the correct ending
        label = int(ex['label'])
        ending = ex['endings'][label]
        ids = build_prompt(enc, ex['ctx'], ending, model.config.block_size)
        mat_sum += depth_frobenius_matrix(model, ids, steps, args.device, dtype, store, reset)

    mat_avg = mat_sum / n

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_name = args.ckpt.replace('/', '_')
    out_path = args.output or f"attn_frobenius_hellaswag_{ckpt_name}_{timestamp}.png"
    npy_path = out_path.rsplit('.', 1)[0] + ".npy"

    title = f"{args.ckpt} (HellaSwag)  |  {n} examples, {args.num_loops} loops x {model.config.n_layer} layers"
    plot_depth_matrix(mat_avg, model.config.n_layer, args.num_loops, title, out_path)
    np.save(npy_path, mat_avg)

    print(f"Saved plot to {out_path}")
    print(f"Saved raw matrix to {npy_path}")


if __name__ == '__main__':
    main()
