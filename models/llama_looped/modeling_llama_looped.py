"""Looped (recurrent-depth) Llama compatible with HuggingFace Llama weights.

Reuses a standard Llama checkpoint (e.g. ``meta-llama/Llama-3.2-3B-Instruct``)
and runs the decoder stack ``num_loops`` times with shared weights. This is the
HuggingFace-side counterpart of vLLM's ``llama_looped.py`` and is meant as a
reference / small-scale eval implementation (vLLM is faster for full runs).

Load with weight reuse::

    from modeling_llama_looped import LlamaLoopedForCausalLM
    model = LlamaLoopedForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct", num_loops=4, recur_mode="latent",
    )

Each ``(loop, layer)`` gets a unique KV-cache slot, so generation with a cache
stays correct across loops.
"""

from typing import Any

import torch
from transformers import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel

from .configuration_llama_looped import LlamaLoopedConfig


def _loop_adapter_prehook(module, args, kwargs):
    """Activate this loop's PEFT (LoRA) adapter before the decoder layer runs.

    The adapter name arrives as a ``loop_adapter=`` kwarg so that, under gradient
    checkpointing, it is captured in the checkpointed partial and re-applied on
    the backward-pass replay — otherwise the replay would run with whatever
    adapter the *last* loop left active and produce wrong gradients.
    """
    adapter = kwargs.pop("loop_adapter", None)
    if adapter is not None:
        for sub in module.modules():
            # Duck-typed peft.BaseTunerLayer (avoids a hard peft dependency).
            # Assign _active_adapter directly instead of set_adapter(): the
            # latter sets requires_grad=False on the other adapters, which
            # silently drops the gradients of every loop but the last one.
            if hasattr(sub, "set_adapter") and hasattr(sub, "_active_adapter"):
                if sub._active_adapter != [adapter]:
                    sub._active_adapter = [adapter]
    return args, kwargs


def _parse_skip_layers(spec: Any) -> dict[int, set[int]]:
    """Normalize a skip spec into ``{loop_idx: {layer_idx, ...}}``.

    Accepts a mapping ``{loop_idx: layer_idx | [layer_idx, ...]}`` (JSON string
    keys allowed) or an iterable of ``(loop_idx, layer_idx)`` pairs.
    """
    result: dict[int, set[int]] = {}
    if not spec:
        return result
    if isinstance(spec, dict):
        for loop_idx, layers in spec.items():
            if isinstance(layers, int):
                layers = [layers]
            result.setdefault(int(loop_idx), set()).update(int(x) for x in layers)
    else:
        for loop_idx, layer_idx in spec:
            result.setdefault(int(loop_idx), set()).add(int(layer_idx))
    return result


class LlamaLoopedModel(LlamaModel):
    config_class = LlamaLoopedConfig

    def __init__(self, config: LlamaConfig):
        """
        collect_loop_hiddens:
            When True, forward() stores the final-norm'ed hidden state of every 
            loop in self.loop_hiddens (used for per-loop deep supervision).
        """
        super().__init__(config)
        self.num_loops = getattr(config, "num_loops", 1)
        self.recur_mode = getattr(config, "recur_mode", "latent")
        assert self.recur_mode in ("latent", "latent_renorm", "token", "soft"), (
            f"Unknown recur_mode: {self.recur_mode!r}"
        )
        self.skip_layers = _parse_skip_layers(getattr(config, "skip_layers", None))
        # Optional per-loop PEFT adapter names (see set_loop_adapters).
        self.loop_adapters: list[str] | None = None
        self._loop_adapter_hooks_registered = False
        
        self.collect_loop_hiddens = False
        self.loop_hiddens: list[torch.Tensor] | None = None
        # Unembedding head for token/soft modes, set by the parent module and
        # kept in a list so it is not registered as a submodule here.
        self._unembed: list[torch.nn.Module] = []

    def set_skip_layers(self, spec: Any) -> None:
        self.skip_layers = _parse_skip_layers(spec)

    def set_loop_adapters(self, adapter_names: list[str] | None) -> None:
        """Use a different PEFT (LoRA) adapter for each loop iteration.

        ``adapter_names[i]`` is activated while loop ``i`` runs. Requires the
        model to be wrapped with peft and one adapter added per loop. Pass
        ``None`` to fall back to the globally active adapter (shared loops).
        """
        if adapter_names is None:
            self.loop_adapters = None
            return
        adapter_names = list(adapter_names)
        if len(adapter_names) != self.num_loops:
            raise ValueError(
                f"need one adapter per loop ({self.num_loops}), got "
                f"{len(adapter_names)}: {adapter_names}"
            )
        self.loop_adapters = adapter_names
        if not self._loop_adapter_hooks_registered:
            for layer in self.layers:
                layer.register_forward_pre_hook(_loop_adapter_prehook, with_kwargs=True)
            self._loop_adapter_hooks_registered = True

    def set_unembedding(self, head: torch.nn.Module) -> None:
        self._unembed = [head]

    def _reembed(self, normed: torch.Tensor) -> torch.Tensor:
        """Decode ``normed`` and produce the next loop's input embedding."""
        head = self._unembed[0] if self._unembed else self.embed_tokens
        logits = head(normed)
        if self.recur_mode == "token":
            return self.embed_tokens(logits.argmax(dim=-1))
        # "soft": expected embedding P @ E (softmax in fp32 for stability).
        probs = torch.softmax(logits.float(), dim=-1).to(normed.dtype)
        return probs @ self.embed_tokens.weight

    @staticmethod
    def _renorm(hidden_states: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Rescale each token's hidden state to per-token L2 norm ``target``."""
        cur = hidden_states.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return hidden_states / cur * target

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        # We route each (loop, layer) to its own KV-cache slot, so the cache must
        # hold num_loops * num_layers slots. A cache pre-sized by generate() to
        # num_hidden_layers has growth disabled (layer_class_to_replicate=None);
        # re-enable it so update() lazily appends the extra slots.
        if (
            isinstance(past_key_values, DynamicCache)
            and past_key_values.layer_class_to_replicate is None
        ):
            past_key_values.layer_class_to_replicate = DynamicLayer
        if position_ids is None:
            past = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0) + past

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)

        num_layers = self.config.num_hidden_layers
        layers = list(self.layers[:num_layers])
        base_idx = [layer.self_attn.layer_idx for layer in layers]
        last = self.num_loops - 1
        hidden_states = inputs_embeds
        # Per-token embedding norm, used to rescale between loops in
        # "latent_renorm" (keeps each loop's input at the embedding magnitude).
        embed_norm = inputs_embeds.norm(dim=-1, keepdim=True)
        self.loop_hiddens = [] if self.collect_loop_hiddens else None
        try:
            for current_loop in range(self.num_loops):
                skip = self.skip_layers.get(current_loop, set())
                loop_kwargs = kwargs
                if self.loop_adapters is not None:
                    # Consumed by _loop_adapter_prehook before the layer forward.
                    loop_kwargs = {
                        **kwargs,
                        "loop_adapter": self.loop_adapters[current_loop],
                    }
                for i, layer in enumerate(layers):
                    if i in skip:
                        continue
                    # Route this (loop, layer) to its own KV-cache slot.
                    layer.self_attn.layer_idx = current_loop * num_layers + i
                    hidden_states = layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_embeddings=position_embeddings,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        **loop_kwargs,
                    )
                if self.loop_hiddens is not None:
                    self.loop_hiddens.append(self.norm(hidden_states))
                if current_loop == last:
                    continue
                if self.recur_mode in ("token", "soft"):
                    # Discrete/soft bottleneck: decode and re-embed for next loop.
                    hidden_states = self._reembed(self.norm(hidden_states))
                elif self.recur_mode == "latent_renorm":
                    # Rescale the residual stream back to the embedding magnitude.
                    hidden_states = self._renorm(hidden_states, embed_norm)
        finally:
            # Restore original per-layer indices.
            for layer, idx in zip(layers, base_idx):
                layer.self_attn.layer_idx = idx

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            # Per-loop normed hiddens ride in the output (not just on self) so
            # they survive DataParallel, whose forward runs on module replicas.
            hidden_states=(
                tuple(self.loop_hiddens) if self.loop_hiddens is not None else None
            ),
        )


class LlamaLoopedForCausalLM(LlamaForCausalLM):
    config_class = LlamaLoopedConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaLoopedModel(config)
        # Give the model the head for token/soft recurrence.
        self.model.set_unembedding(self.lm_head)
        self.post_init()

    def set_skip_layers(self, spec: Any) -> None:
        self.model.set_skip_layers(spec)

    def set_loop_adapters(self, adapter_names: list[str] | None) -> None:
        self.model.set_loop_adapters(adapter_names)
