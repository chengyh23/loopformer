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
    """

    model_type = "llama_looped"

    def __init__(
        self,
        num_loops: int = 1,
        recur_mode: str = "latent",
        skip_layers=None,
        **kwargs,
    ):
        self.num_loops = num_loops
        self.recur_mode = recur_mode
        self.skip_layers = skip_layers
        super().__init__(**kwargs)
