import math
import pytest
import torch
import torch.nn.functional as F

from promptguard.core import Config, Router, block_mean, grounding, importance, select_pattern, sparse_attention


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(7)


@pytest.mark.parametrize("n,bs", [(1, 4), (7, 3), (8, 2)])
def test_block_mean_and_equation2(n, bs):
    q, k = torch.randn(2, 3, n, 5), torch.randn(2, 3, n, 5)
    qb = torch.stack([q[..., i:i+bs, :].mean(-2) for i in range(0, n, bs)], -2)
    kb = torch.stack([k[..., i:i+bs, :].mean(-2) for i in range(0, n, bs)], -2)
    torch.testing.assert_close(block_mean(q, bs), qb)
    expected = (qb @ kb.transpose(-1, -2) / math.sqrt(5)).softmax(-1)
    torch.testing.assert_close(importance(q, k, bs), expected)


def test_grounding_independent_normalization_and_mean_logits():
    q, k = torch.randn(1, 2, 7, 4), torch.randn(1, 2, 6, 4)
    units = [[0, 2], [1], [3, 4, 5]]
    g = grounding(q, k, units, 3)
    qb = torch.stack([q[..., i:i+3, :].mean(-2) for i in range(0, 7, 3)], -2)
    logits = qb @ k.transpose(-1, -2) / 2
    expected = torch.stack([logits[..., unit].mean(-1).softmax(-1) for unit in units], -2)
    torch.testing.assert_close(g, expected)
    torch.testing.assert_close(g.sum(-1), torch.ones(1, 2, 3))


@pytest.mark.parametrize("density", [0.001, 0.1, 0.4, 1.0])
@pytest.mark.parametrize("alpha", [0, 0.2, 1])
def test_exact_budget_and_all_semantic_quotas(density, alpha):
    p = torch.randn(5, 5).softmax(-1)
    g = torch.randn(3, 5).softmax(-1)
    cfg = Config(density=density, alpha=alpha)
    mask, protected, quota = select_pattern(p, g, cfg)
    budget = math.floor(density * 25)
    assert int(mask.sum()) == budget
    assert int(protected.sum()) <= 3 * quota <= alpha * budget
    assert not (protected & ~mask).any()
    for gm in g:
        scores = (gm[:, None] * p).flatten()
        ranked = sorted(range(25), key=lambda i: (-float(scores[i]), i))[:quota]
        assert mask.flatten()[ranked].all()
    available = sorted([i for i in range(25) if not protected.flatten()[i]], key=lambda i: (-float(p.flatten()[i]), i))
    assert mask.flatten()[available[:budget-int(protected.sum())]].all()


def test_overlapping_protection_refills_and_small_subject_is_protected():
    p = torch.tensor([[.6, .4, 0.], [.55, .45, 0.], [.34, .33, .33]])
    g = torch.tensor([[0., 0., 1.], [0., 0., 1.]])
    mask, protected, quota = select_pattern(p, g, Config(density=0.5, alpha=0.5))
    assert quota == 1 and protected.sum() == 1 and mask.sum() == 4
    assert mask[2, 0]  # global top-4 alone would miss this small subject


def dense_oracle(q, k, v, patterns, bs, text_k=None, text_v=None):
    n = q.shape[-2]
    mask = torch.stack([torch.stack([patterns[b, h].mask.repeat_interleave(bs, 0).repeat_interleave(bs, 1)[:n, :n]
                                   for h in range(q.shape[1])]) for b in range(q.shape[0])])
    if text_k is not None:
        mask = torch.cat((torch.ones((*mask.shape[:-1], text_k.shape[-2]), device=q.device, dtype=torch.bool), mask), -1)
        k, v = torch.cat((text_k, k), -2), torch.cat((text_v, v), -2)
    logits = q.double() @ k.double().transpose(-1, -2) / math.sqrt(q.shape[-1])
    probs = logits.masked_fill(~mask, -torch.inf).softmax(-1).nan_to_num()
    return (probs @ v.double()).to(q.dtype)


@pytest.mark.parametrize("density", [0.001, 0.4, 1.])
@pytest.mark.parametrize("joint", [False, True])
def test_sparse_matches_independent_dense_oracle(density, joint):
    q, k, v = (torch.randn(2, 2, 7, 4) for _ in range(3))
    kt, vt = torch.randn(2, 2, 3, 4), torch.randn(2, 2, 3, 4)
    r = Router(Config(density=density, block_size=3, warmup_steps=0))
    patterns = r.route(q, k, q, kt, [[0], [1, 2]], 0)
    tk, tv = (kt, vt) if joint else (None, None)
    actual = sparse_attention(q, k, v, patterns, 3, tk, tv)
    torch.testing.assert_close(actual, dense_oracle(q, k, v, patterns, 3, tk, tv), atol=1e-6, rtol=1e-5)
    assert torch.isfinite(actual).all()


def test_reuse_avoids_importance_and_keeps_reference(monkeypatch):
    import promptguard.core as core
    q, k = torch.randn(1, 1, 4, 2), torch.randn(1, 1, 4, 2)
    kt = torch.randn(1, 1, 2, 2)
    # Single moving requirement amongst stable ones: max, not mean, triggers.
    initial = torch.tensor([[[[.99, .01], [.99, .01], [.99, .01]]]])
    near = torch.tensor([[[[.8, .2], [.99, .01], [.99, .01]]]])
    moved = torch.tensor([[[[.01, .99], [.99, .01], [.99, .01]]]])
    values = iter([initial, initial, near, moved])
    monkeypatch.setattr(core, "grounding", lambda *args: next(values))
    r = Router(Config(block_size=2, warmup_steps=1, delta=.5))
    assert r.route(q, k, q, kt, [[0], [1], [0, 1]], 0) is None
    r.route(q, k, q, kt, [[0], [1], [0, 1]], 1)
    old = r.cache[0, 0]
    for step in (2, 3):
        r.route(q * 10, k * 10, q, kt, [[0], [1], [0, 1]], step)
        assert r.cache[0, 0] is old
        torch.testing.assert_close(old.reference, initial[0, 0])
    assert r.importance_evaluations == 1
    r.route(q, k, q, kt, [[0], [1], [0, 1]], 4)
    assert r.cache[0, 0] is not old and r.cache[0, 0].refreshed_at == 4
    assert r.importance_evaluations == 2


def test_threshold_equality_reuses(monkeypatch):
    import promptguard.core as core
    q = torch.ones(1, 1, 4, 2)
    g = torch.ones(1, 1, 1, 2) / 2
    monkeypatch.setattr(core, "grounding", lambda *args: g)
    exact_distance = float((1 - F.cosine_similarity(g, g, dim=-1)).clamp(0, 1).max())
    r = Router(Config(block_size=2, warmup_steps=0, delta=exact_distance))
    r.route(q, q, q, q, [[0]], 0)
    r.route(q, q, q, q, [[0]], 1)
    assert r.importance_evaluations == 1


def test_reset_and_head_sample_isolation():
    q = torch.randn(2, 3, 8, 4)
    r = Router(Config(block_size=2, warmup_steps=0))
    r.route(q, q, q, q, [[0]], 0)
    assert len(r.cache) == 6
    assert len({p.reference.data_ptr() for p in r.cache.values()}) == 6
    r.reset()
    assert not r.cache and not r.events
    r.route(q, q, q, q, [[1]], 0)
    assert r.importance_evaluations == 6


@pytest.mark.parametrize("kwargs", [{"density": 0}, {"density": 1.1}, {"alpha": -1}, {"delta": 2}, {"block_size": 0}, {"warmup_steps": -1}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


@pytest.mark.parametrize("units", [[], [[]], [[-1]], [[5]]])
def test_bad_units(units):
    q = torch.randn(1, 1, 4, 2)
    with pytest.raises(ValueError):
        grounding(q, q, units, 2)


def cuda_arch_skip_reason():
    """Skip only when this torch build has no kernel image for the visible GPU."""
    if not torch.cuda.is_available():
        return "requires CUDA"
    major, minor = torch.cuda.get_device_capability()
    required = f"sm_{major}{minor}"
    compiled = torch.cuda.get_arch_list()
    if required not in compiled:
        return (f"GPU compute capability {required} is not supported by the installed "
                f"torch {torch.__version__} build, which supports {', '.join(compiled)}")
    return None


CUDA_ARCH_SKIP_REASON = cuda_arch_skip_reason()


@pytest.mark.skipif(CUDA_ARCH_SKIP_REASON is not None, reason=CUDA_ARCH_SKIP_REASON or "")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_half_precision(dtype):
    q, k, v = [torch.randn(1, 2, 17, 8, device="cuda", dtype=dtype) for _ in range(3)]
    r = Router(Config(block_size=4, warmup_steps=0))
    p = r.route(q, k, q, k, [[0, 1], [3]], 0)
    y = sparse_attention(q, k, v, p, 4)
    torch.testing.assert_close(y, dense_oracle(q, k, v, p, 4), atol=.02, rtol=.02)
