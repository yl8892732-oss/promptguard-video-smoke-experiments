import pytest
import torch

pytest.importorskip("diffusers")
from diffusers import CogVideoXTransformer3DModel
from promptguard import Config
from promptguard.cogvideox import Controller


def tiny_model():
    return CogVideoXTransformer3DModel(num_attention_heads=2, attention_head_dim=8,
        in_channels=4, out_channels=4, time_embed_dim=16, text_embed_dim=12,
        num_layers=2, sample_width=8, sample_height=8, sample_frames=5,
        patch_size=2, max_text_seq_length=6).eval()


@pytest.mark.parametrize("cfg", [False, True])
def test_real_diffusers_transformer_dense_equivalence_and_sparse_steps(cfg):
    torch.manual_seed(42)
    model = tiny_model()
    batch = 2 if cfg else 1
    x = torch.randn(batch, 2, 4, 8, 8)
    text = torch.randn(batch, 6, 12)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.inference_mode():
        baseline = model(x, text, timestep=torch.tensor([10])).sample
        ctrl = Controller(model, [[0, 1], [2, 3]], Config(density=1, block_size=4, warmup_steps=0), cfg)
        dense = model(x, text, timestep=torch.tensor([10])).sample
        torch.testing.assert_close(dense, baseline, atol=1e-6, rtol=1e-5)
        ctrl.restore()
        ctrl = Controller(model, [[0, 1], [2, 3]], Config(block_size=4, warmup_steps=1, delta=1), cfg)
        for i in range(4):
            ctrl.set_step(i)
            x = model(x, text, timestep=torch.tensor([10-i])).sample
            assert x.shape == (batch, 2, 4, 8, 8) and torch.isfinite(x).all()
        for proc in ctrl.processors.values():
            assert proc.router.importance_evaluations == 2
            assert proc.router.events[-1]["reused"] == 2
            assert len(proc.router.cache) == 2
        ctrl.restore()
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name], atol=0, rtol=0)


def test_actual_pipeline_scheduler_callback_and_vae_decode():
    """Offline test with small random weights; distinct from pretrained results."""
    import numpy as np
    from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler, CogVideoXPipeline
    vae = AutoencoderKLCogVideoX(block_out_channels=(8, 8), latent_channels=4,
        down_block_types=("CogVideoXDownBlock3D",) * 2,
        up_block_types=("CogVideoXUpBlock3D",) * 2,
        layers_per_block=1, norm_num_groups=4, temporal_compression_ratio=2,
        sample_height=16, sample_width=16).eval()
    model = CogVideoXTransformer3DModel(num_attention_heads=2, attention_head_dim=8,
        in_channels=4, out_channels=4, time_embed_dim=16, text_embed_dim=12,
        num_layers=2, sample_width=8, sample_height=8, sample_frames=5,
        patch_size=2, temporal_compression_ratio=2, max_text_seq_length=6).eval()
    pipe = CogVideoXPipeline(tokenizer=None, text_encoder=None, vae=vae,
        transformer=model, scheduler=CogVideoXDDIMScheduler())
    ctrl = Controller(model, [[0, 1], [2, 3]], Config(block_size=4, delta=1), cfg=True)
    embeds = torch.randn(1, 6, 12)
    with torch.inference_mode():
        result = pipe(prompt_embeds=embeds, negative_prompt_embeds=torch.zeros_like(embeds),
            height=16, width=16, num_frames=5, num_inference_steps=3, guidance_scale=6,
            max_sequence_length=6, callback_on_step_end=ctrl.callback, output_type="np").frames
    assert result.shape == (1, 5, 16, 16, 3) and np.isfinite(result).all()
    for processor in ctrl.processors.values():
        assert [e["step"] for e in processor.router.events] == [0, 1, 2]
        assert processor.router.events[0]["warmup"]
        assert processor.router.events[1]["refreshed"] == 2
        assert processor.router.events[2]["reused"] == 2
