"""Config for the looped (recurrent-depth) Llama, reusing Llama weights."""

from transformers import LlamaConfig


class LlamaLoopedConfig(LlamaConfig):
    """Llama config plus loop controls.

    Args:
        num_loops: Number of times the decoder stack is run (1 == vanilla Llama).
        recur_mode: How state passes between loops:
            "latent" - hidden states flow directly (continuous residual);
            "token"  - decode->argmax->re-embed (hard bottleneck);
            "soft"   - next input = softmax(logits) @ E (soft bottleneck).
        skip_layers: Optional {loop_idx: layer_idx | [layer_idx, ...]} of layers
            to skip per loop (JSON string keys allowed).
        use_loop_sandwichnorm: If True, each decoder layer uses a 4-norm
            sandwich layout (ln_attn_inner/post_attn_ln/ln_mlp_inner/
            post_mlp_ln) instead of the standard 2-norm pre-norm layout.
        use_loop_attn_residual: If True, apply Attention Residuals (Moonshot
            AttnRes) over the unrolled loop-depth: each layer's input becomes a
            learned softmax attention over all prior layer outputs (+ embedding),
            and the LM-head input is a readout attention over all of them. The
            inter-loop recur_mode carry is subsumed; requires a latent-family
            recur_mode. Newly-initialized weights, trained fully and saved as
            loop_attn_res.pt (not in the LoRA adapter).
        loop_attn_res_block: Reserved for a memory-efficient block-AttnRes
            variant (0 == full AttnRes). Not yet used.
    """

    model_type = "llama_looped"

    def __init__(
        self,
        num_loops: int = 1,
        recur_mode: str = "latent",
        skip_layers=None,
        use_loop_sandwichnorm: bool = False,
        use_loop_attn_residual: bool = False,
        loop_attn_res_block: int = 0,
        **kwargs,
    ):
        self.num_loops = num_loops
        self.recur_mode = recur_mode
        self.skip_layers = skip_layers
        self.use_loop_sandwichnorm = use_loop_sandwichnorm
        self.use_loop_attn_residual = use_loop_attn_residual
        self.loop_attn_res_block = loop_attn_res_block
        super().__init__(**kwargs)
