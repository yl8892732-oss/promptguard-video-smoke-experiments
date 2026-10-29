"""PromptGuard equations (2)--(9), independent of a diffusion model.

Routing is inference-only. Tensors have shape [batch, heads, tokens, dim].
The reference sparse executor gathers only selected key blocks; it never builds
a dense token-by-token attention mask. This is not an optimized CUDA kernel.
"""
from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    density: float = 0.4
    alpha: float = 0.2
    delta: float = 0.05
    block_size: int = 128
    warmup_steps: int = 1

    def __post_init__(self):
        if not 0 < self.density <= 1:
            raise ValueError("density must be in (0, 1]")
        if not 0 <= self.alpha <= 1 or not 0 <= self.delta <= 1:
            raise ValueError("alpha and delta must be in [0, 1]")
        if self.block_size < 1 or self.warmup_steps < 0:
            raise ValueError("invalid block_size or warmup_steps")


def block_mean(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Mean over contiguous blocks, excluding padding in the last block."""
    n, d = x.shape[-2:]
    if n == 0 or block_size < 1:
        raise ValueError("empty sequence or invalid block size")
    blocks = math.ceil(n / block_size)
    padded = F.pad(x.float(), (0, 0, 0, blocks * block_size - n))
    sums = padded.reshape(*x.shape[:-2], blocks, block_size, d).sum(-2)
    counts = torch.full((blocks,), block_size, device=x.device, dtype=torch.float32)
    counts[-1] = n - (blocks - 1) * block_size
    return sums / counts[:, None]


def importance(q: torch.Tensor, k: torch.Tensor, block_size: int) -> torch.Tensor:
    """Equation (2): normalization over key blocks, per query block."""
    qb, kb = block_mean(q, block_size), block_mean(k, block_size)
    return (qb @ kb.transpose(-1, -2) / math.sqrt(q.shape[-1])).softmax(-1)


def grounding(q_cross, k_text, units: Sequence[Sequence[int]], block_size: int):
    """Equation (3): each semantic unit independently normalizes over blocks."""
    if not units:
        raise ValueError("at least one nonempty semantic unit is required")
    for unit in units:
        if not unit or min(unit) < 0 or max(unit) >= k_text.shape[-2]:
            raise ValueError("semantic unit contains empty/out-of-range token indices")
    keys = torch.stack([k_text[..., list(unit), :].float().mean(-2) for unit in units], -2)
    qb = block_mean(q_cross, block_size)
    return (keys @ qb.transpose(-1, -2) / math.sqrt(q_cross.shape[-1])).softmax(-1)


def _top_indices(scores, count):
    # Stable ties are reproducible: lower flattened block-pair index wins.
    return torch.argsort(scores.flatten(), descending=True, stable=True)[:count]


def select_pattern(p: torch.Tensor, g: torch.Tensor, config: Config):
    """Equations (4)--(7), for one sample and one head.

    No row-completion edges are added: the budget is EXACTLY floor(rho * B²).
    Global Top-K may leave rows empty. The executor returns zero for such rows
    in pure self-attention; joint attention can still attend to dense text keys.
    """
    if p.ndim != 2 or p.shape[0] != p.shape[1] or g.ndim != 2 or g.shape[-1] != p.shape[0]:
        raise ValueError("expected P[B,B], G[M,B]")
    if g.shape[0] == 0:
        raise ValueError("no semantic units")
    budget = math.floor(config.density * p.numel())
    quota = math.floor(config.alpha * budget / g.shape[0])
    protected = torch.zeros_like(p, dtype=torch.bool).flatten()
    for gm in g:
        protected[_top_indices(gm[:, None] * p, quota)] = True
    mask = protected.clone()
    remaining = budget - int(protected.sum().item())
    if remaining:
        scores = p.flatten().masked_fill(protected, -torch.inf)
        mask[_top_indices(scores, remaining)] = True
    return mask.reshape_as(p), protected.reshape_as(p), quota


@dataclass
class HeadPattern:
    mask: torch.Tensor
    protected: torch.Tensor
    reference: torch.Tensor
    rows: tuple
    refreshed_at: int
    quota: int


class PatternCache(dict):
    """Cache a compact execution plan (block IDs, never N-by-N token masks)."""
    plan = None


class Router:
    """One instance per layer; independent cache for each sample/head.

    Call reset() for every new generation/prompt, even if shapes are unchanged.
    Grounding references change ONLY on refresh, never on reuse.
    """
    def __init__(self, config=Config()):
        self.config = config
        self.reset()

    def reset(self):
        self.cache = PatternCache()
        self.signature = None
        self.last_step = -1
        self.events = []
        self.importance_evaluations = 0

    @torch.no_grad()
    def route(self, q, k, q_cross, k_text, units, step):
        if q.ndim != 4 or q.shape != k.shape or q_cross.shape[:3] != q.shape[:3]:
            raise ValueError("Q/K must match [batch,heads,N,d]; cross queries must share batch,head,N")
        if k_text.shape[:2] != q.shape[:2] or k_text.shape[-1] != q_cross.shape[-1]:
            raise ValueError("cross attention query/key dimensions do not match")
        if step < 0:
            raise ValueError("step is the nonnegative denoising loop index")
        signature = (tuple(q.shape), tuple(q_cross.shape), tuple(k_text.shape), str(q.device),
                     str(q.dtype), tuple(tuple(u) for u in units))
        if self.signature != signature or step < self.last_step:
            self.reset()
        self.signature, self.last_step = signature, step
        if step < self.config.warmup_steps:
            self.events.append(dict(step=step, warmup=True, refreshed=0, reused=0))
            return None
        g = grounding(q_cross, k_text, units, self.config.block_size)
        refreshed = reused = 0
        max_drift = 0.0
        for b in range(q.shape[0]):
            for h in range(q.shape[1]):
                old = self.cache.get((b, h))
                drift = (1 - F.cosine_similarity(g[b, h], old.reference, dim=-1)).clamp(0, 1).max().item() if old else None
                if drift is not None:
                    max_drift = max(max_drift, drift)
                if old is None or drift > self.config.delta:
                    p = importance(q[b, h], k[b, h], self.config.block_size)
                    self.importance_evaluations += 1
                    mask, protected, quota = select_pattern(p, g[b, h], self.config)
                    # Synchronize once on refresh; reuse Python block lists thereafter.
                    rows = tuple(tuple(row.nonzero()[0].tolist()) for row in mask.cpu().numpy())
                    self.cache[b, h] = HeadPattern(mask, protected, g[b, h].clone(), rows, step, quota)
                    self.cache.plan = None
                    refreshed += 1
                else:
                    reused += 1
        self.events.append(dict(step=step, warmup=False, refreshed=refreshed, reused=reused,
                                max_drift=max_drift))
        return self.cache


def sparse_attention(q, k, v, patterns, block_size, text_k=None, text_v=None):
    """Exact softmax over selected visual keys plus optional ALL text keys.

    Never normalizes text and video attention separately. This matters for
    CogVideoX's joint attention. Empty pure-visual rows are explicitly zero.
    """
    if q.shape != k.shape or k.shape[:3] != v.shape[:3]:
        raise ValueError("incompatible Q/K/V shapes")
    if (text_k is None) != (text_v is None):
        raise ValueError("text_k/text_v must be supplied together")
    if patterns is None:
        kk = torch.cat((text_k, k), -2) if text_k is not None else k
        vv = torch.cat((text_v, v), -2) if text_v is not None else v
        return F.scaled_dot_product_attention(q, kk, vv)
    out = torch.zeros((*q.shape[:-1], v.shape[-1]), dtype=v.dtype, device=v.device)
    n = q.shape[-2]
    heads = q.shape[1]
    # Batch heads with equally many selected keys. Group tail-block membership
    # separately so no padding / unwanted video block pair needs to be computed.
    plan = getattr(patterns, "plan", None)
    if plan is None:
        plan = []
        blocks = math.ceil(n / block_size)
        for u in range(blocks):
            groups = {}
            for b in range(q.shape[0]):
                for h in range(heads):
                    row = patterns[b, h].rows[u]
                    tail = bool(n % block_size and row and row[-1] == blocks - 1)
                    groups.setdefault((len(row), tail), []).append((b*heads+h, row))
            for (count, tail), members in groups.items():
                ids = torch.tensor([m[0] for m in members], device=q.device)
                indices = torch.tensor([m[1] for m in members], dtype=torch.long, device=q.device).reshape(len(members), count)
                trim = block_size - n % block_size if tail else 0
                plan.append((u, ids, indices, trim))
        if isinstance(patterns, PatternCache):
            patterns.plan = plan
    qf, kf, vf, of = [x.reshape(-1, *x.shape[-2:]) for x in (q, k, v, out)]
    tkf = text_k.reshape(-1, *text_k.shape[-2:]) if text_k is not None else None
    tvf = text_v.reshape(-1, *text_v.shape[-2:]) if text_v is not None else None
    offsets = torch.arange(block_size, device=q.device)
    for u, ids, blocks, trim in plan:
        if blocks.shape[-1] == 0 and text_k is None:
            continue
        indices = (blocks[..., None] * block_size + offsets).flatten(1)
        if trim:
            indices = indices[:, :-trim]
        kk, vv = kf[ids[:, None], indices], vf[ids[:, None], indices]
        if tkf is not None:
            kk, vv = torch.cat((tkf[ids], kk), 1), torch.cat((tvf[ids], vv), 1)
        start, end = u*block_size, min((u+1)*block_size, n)
        # Explicit 4D shape allows CUDA Flash SDPA dispatch.
        of[ids, start:end] = F.scaled_dot_product_attention(
            qf[ids, start:end].unsqueeze(1), kk.unsqueeze(1), vv.unsqueeze(1)).squeeze(1)
    return out
