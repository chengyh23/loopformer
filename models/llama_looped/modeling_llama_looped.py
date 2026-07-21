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
from torch import nn

from transformers import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel, LlamaDecoderLayer, LlamaAttention, LlamaMLP, LlamaRMSNorm
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

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

class LlamaLoopedDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)

        self.use_loop_sandwichnorm = bool(getattr(config, "use_loop_sandwichnorm", False))
        if self.use_loop_sandwichnorm:
            self.ln_attn_inner = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.ln_mlp_inner = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attn_ln = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_mlp_ln = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.input_layernorm = None
            self.post_attention_layernorm = None
        else:
            self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if self.use_loop_sandwichnorm:
            residual = hidden_states
            hidden_states = self.ln_attn_inner(hidden_states)
            # Self Attention
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + hidden_states
            hidden_states = self.post_attn_ln(hidden_states)
            # Fully Connected
            residual = hidden_states
            hidden_states = self.ln_mlp_inner(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            hidden_states = self.post_mlp_ln(hidden_states)
            return hidden_states
        
        # Standard pre-norm path
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


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
        self.layers = nn.ModuleList(
            [LlamaLoopedDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

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

    def _block_forward(self, embeds, attention_mask_2d, position_ids, loop_adapter=None):
        """One pass of the full decoder stack over ``embeds`` (no KV cache)."""
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=embeds,
            attention_mask=attention_mask_2d,
            past_key_values=None,
            position_ids=position_ids,
        )
        position_embeddings = self.rotary_emb(embeds, position_ids=position_ids)
        extra = {} if loop_adapter is None else {"loop_adapter": loop_adapter}
        hidden = embeds
        for layer in self.layers[: self.config.num_hidden_layers]:
            hidden = layer(
                hidden,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                use_cache=False,
                **extra,
            )
        return hidden

    def forward_latent_branches(self, input_ids, attention_mask, branch_ids, branch_mask):
        """Latent-carry forward with a per-loop generative readout branch.

        Loop ``i`` runs the stack over ``[latent(Q); embed(branch_i)]``: the
        teacher-forced branch text (e.g. CoT step ``i``) reads Q's loop-``i``
        representation through causal attention, but only the Q positions are
        carried to loop ``i+1`` — branch text never enters later loops' token
        stream, so the only inter-loop channel is the latent.

        Args:
            input_ids:      (B, Lq) right-padded question tokens.
            attention_mask: (B, Lq) 1 on real question tokens.
            branch_ids:     (B, num_loops, Lb) right-padded branch tokens.
            branch_mask:    (B, num_loops, Lb) 1 on real branch tokens.

        Returns a list with one normed readout (B, Lb, H) per loop, aligned so
        that readout position ``j`` predicts branch token ``j`` (position 0 is
        read from the last real Q token of each example).
        """
        assert self.recur_mode in ("latent", "latent_renorm"), (
            "forward_latent_branches supports latent carry only, got "
            f"{self.recur_mode!r}"
        )
        assert branch_ids.shape[1] == self.num_loops
        bsz, q_len_padded = input_ids.shape
        device = input_ids.device
        q_lens = attention_mask.sum(dim=1)  # (B,)
        q_pos = (
            torch.arange(q_len_padded, device=device).unsqueeze(0).expand(bsz, -1)
        )
        batch_idx = torch.arange(bsz, device=device)
        h = self.embed_tokens(input_ids)
        embed_norm = h.norm(dim=-1, keepdim=True)
        readouts = []
        for i in range(self.num_loops):
            b_ids = branch_ids[:, i]
            lb = b_ids.shape[1]
            embeds = torch.cat([h, self.embed_tokens(b_ids)], dim=1)
            mask2d = torch.cat([attention_mask, branch_mask[:, i]], dim=1)
            # Branch positions continue each example's real question length, so
            # RoPE offsets are correct despite right padding (pad positions may
            # collide with branch positions but are attention-masked anyway).
            b_pos = q_lens.unsqueeze(1) + torch.arange(lb, device=device).unsqueeze(0)
            pos = torch.cat([q_pos, b_pos], dim=1)
            adapter = self.loop_adapters[i] if self.loop_adapters is not None else None
            out = self._block_forward(embeds, mask2d, pos, loop_adapter=adapter)
            # Readout position j predicts branch token j; position 0 reads from
            # the last real Q token.
            last_q = out[batch_idx, q_lens - 1].unsqueeze(1)
            readout = torch.cat(
                [last_q, out[:, q_len_padded : q_len_padded + lb - 1]], dim=1
            )
            readouts.append(self.norm(readout))
            if i == self.num_loops - 1:
                break
            h = out[:, :q_len_padded]
            if self.recur_mode == "latent_renorm":
                h = self._renorm(h, embed_norm)
        return readouts

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

    def forward_latent_branches(self, *args, **kwargs):
        return self.model.forward_latent_branches(*args, **kwargs)

    @torch.no_grad()
    def generate_latent(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 64,
        eos_token_id: int | None = None,
    ) -> torch.LongTensor:
        """Greedy decoding matching forward_latent_branches training.

        Runs loops 0..N-2 over the question only (latent carry, no branch),
        then generates the answer with single final-loop passes over
        ``[latent(Q); generated tokens]`` — no CoT tokens are emitted and the
        intermediate loops run exactly once regardless of answer length.

        Returns only the generated tokens, (B, <= max_new_tokens).
        """
        m = self.model
        bsz, q_len = input_ids.shape
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        q_lens = attention_mask.sum(dim=1)
        q_pos = torch.arange(q_len, device=device).unsqueeze(0).expand(bsz, -1)
        embed_norm = None
        h = m.embed_tokens(input_ids)
        if m.recur_mode == "latent_renorm":
            embed_norm = h.norm(dim=-1, keepdim=True)
        adapters = m.loop_adapters or [None] * m.num_loops
        for i in range(m.num_loops - 1):
            h = m._block_forward(h, attention_mask, q_pos, loop_adapter=adapters[i])
            if m.recur_mode == "latent_renorm":
                h = m._renorm(h, embed_norm)

        generated = torch.zeros(bsz, 0, dtype=torch.long, device=device)
        done = torch.zeros(bsz, dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            t = generated.shape[1]
            embeds = torch.cat([h, m.embed_tokens(generated)], dim=1)
            mask2d = torch.cat(
                [attention_mask, torch.ones(bsz, t, dtype=attention_mask.dtype, device=device)],
                dim=1,
            )
            g_pos = q_lens.unsqueeze(1) + torch.arange(t, device=device).unsqueeze(0)
            pos = torch.cat([q_pos, g_pos], dim=1)
            out = m._block_forward(embeds, mask2d, pos, loop_adapter=adapters[-1])
            # Next-token position: last generated token, or last real Q token
            # when nothing has been generated yet.
            if t == 0:
                read = out[torch.arange(bsz, device=device), q_lens - 1]
            else:
                read = out[:, -1]
            logits = self.lm_head(m.norm(read))
            next_tok = logits.argmax(dim=-1)
            if eos_token_id is not None:
                next_tok = torch.where(
                    done, torch.full_like(next_tok, eos_token_id), next_tok
                )
                done = done | (next_tok == eos_token_id)
            generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)
            if eos_token_id is not None and bool(done.all()):
                break
        return generated
