"""Diffusers CogVideoX-2B joint-attention adapter.

Adaptation of Method.md: visual Q and text K from joint attention supply the
cross-modal projections for Eq. (3). Text-query attention stays dense. With CFG,
the negative branch stays dense; positive semantics apply only to the positive
branch. The exact K budget refers to positive video-to-video block interactions.
"""
import torch
import torch.nn.functional as F

from .core import Config, Router, sparse_attention


class CogVideoXPromptGuardProcessor:
    def __init__(self, config, token_units, cfg=False):
        self.router = Router(config)
        self.token_units = token_units
        self.cfg = cfg
        self.step = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states,
                 attention_mask=None, image_rotary_emb=None):
        if attention_mask is not None:
            raise ValueError("this adapter supports the unmasked CogVideoX-2B pipeline only")
        length = encoder_hidden_states.shape[1]
        x = torch.cat((encoder_hidden_states, hidden_states), 1)
        batch = x.shape[0]
        if batch != (2 if self.cfg else 1):
            raise ValueError("adapter supports one prompt / one video; CFG batch must be 2")
        def project(layer):
            return layer(x).reshape(batch, -1, attn.heads, attn.inner_dim // attn.heads).transpose(1, 2)
        q, k, v = project(attn.to_q), project(attn.to_k), project(attn.to_v)
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            q[:, :, length:] = apply_rotary_emb(q[:, :, length:], image_rotary_emb)
            if not attn.is_cross_attention:
                k[:, :, length:] = apply_rotary_emb(k[:, :, length:], image_rotary_emb)
        start = 1 if self.cfg else 0
        qv, kv, vv = q[start:, :, length:], k[start:, :, length:], v[start:, :, length:]
        kt, vt = k[start:, :, :length], v[start:, :, :length]
        patterns = self.router.route(qv, kv, qv, kt, self.token_units, self.step)
        if patterns is None or self.router.config.density == 1:
            result = F.scaled_dot_product_attention(q, k, v)
        else:
            positive_video = sparse_attention(qv, kv, vv, patterns, self.router.config.block_size, kt, vt)
            positive_text = F.scaled_dot_product_attention(q[start:, :, :length], k[start:], v[start:])
            positive = torch.cat((positive_text, positive_video), -2)
            result = torch.cat((F.scaled_dot_product_attention(q[:1], k[:1], v[:1]), positive), 0) if self.cfg else positive
        result = result.transpose(1, 2).reshape(batch, -1, attn.inner_dim).to(x.dtype)
        result = attn.to_out[1](attn.to_out[0](result))
        return result[:, length:], result[:, :length]


class Controller:
    def __init__(self, transformer, token_units, config=Config(), cfg=False):
        self.transformer = transformer
        self.original = dict(transformer.attn_processors)
        self.processors = {name: CogVideoXPromptGuardProcessor(config, token_units, cfg)
                           for name in self.original}
        transformer.set_attn_processor(dict(self.processors))

    def set_step(self, step):
        for processor in self.processors.values():
            processor.step = step

    def reset(self):
        for processor in self.processors.values():
            processor.router.reset()
            processor.step = 0

    def callback(self, pipeline, step_index, timestep, callback_kwargs):
        self.set_step(step_index + 1)
        return callback_kwargs

    def restore(self):
        self.transformer.set_attn_processor(dict(self.original))

    def report(self):
        layers = {}
        for name, processor in self.processors.items():
            router = processor.router
            patterns = list(router.cache.values())
            layers[name] = dict(events=router.events, importance_evaluations=router.importance_evaluations,
                                cached_heads=len(patterns),
                                budgets=[int(p.mask.sum()) for p in patterns],
                                protected=[int(p.protected.sum()) for p in patterns],
                                quotas=[p.quota for p in patterns])
        return layers
