"""CPU smoke tests of real VideoMAE encoder/decoder/classifier paths."""

import argparse
import ast
from pathlib import Path

import pytest
import torch

from modeling_finetune import VisionTransformer
from modeling_pretrain import PretrainVisionTransformer
from video_rope import BASELINE_POS_MODES, apply_video_rope, build_baseline_coordinates


torch.set_num_threads(1)


def pretrain_model(mode, **kwargs):
    # Keep ViT-S encoder/decoder head dimensions (64) with a small smoke grid.
    return PretrainVisionTransformer(
        img_size=32, num_frames=4, patch_size=16, tubelet_size=2,
        encoder_embed_dim=384, encoder_num_heads=6, encoder_depth=2,
        decoder_embed_dim=192, decoder_num_heads=3, decoder_depth=1,
        qkv_bias=True, pos_mode=mode, rope_axis_dims=(24, 24, 16),
        **kwargs,
    )


def finetune_model(mode, **kwargs):
    return VisionTransformer(
        img_size=32, all_frames=4, patch_size=16, tubelet_size=2,
        embed_dim=384, num_heads=6, depth=2, num_classes=51,
        qkv_bias=True, pos_mode=mode, rope_axis_dims=(24, 24, 16),
        init_scale=1.0, **kwargs,
    )


def tube_mask():
    return torch.tensor([[False, True, False, True] * 2] * 2)


def test_vanilla_uses_full_grid_indices():
    coords = build_baseline_coordinates("vanilla_rope", 2, (2, 2, 2), torch.device("cpu"))
    torch.testing.assert_close(coords[0, :, 0], torch.arange(8).float())
    visible = coords[~tube_mask()].reshape(2, -1, 1)
    torch.testing.assert_close(visible[0, :, 0], torch.tensor([0., 2., 4., 6.]))


def test_vanilla_matches_independent_reference():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 8, 64)
    coords = build_baseline_coordinates("vanilla_rope", 2, (2, 2, 2), q.device)
    actual, _ = apply_video_rope(q, q, coords, "vanilla_rope")
    expected = q.clone()
    for pair in range(32):
        angle = coords[..., 0].unsqueeze(1) / (10000.0 ** (2 * pair / 64))
        a, b = q[..., 2 * pair], q[..., 2 * pair + 1]
        expected[..., 2 * pair] = a * angle.cos() - b * angle.sin()
        expected[..., 2 * pair + 1] = a * angle.sin() + b * angle.cos()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("mode", BASELINE_POS_MODES)
def test_rotation_norm_tail_and_gradients(mode):
    q = torch.randn(2, 3, 8, 80, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    coords = build_baseline_coordinates(mode, 2, (2, 2, 2), q.device)
    qr, kr = apply_video_rope(q, k, coords, mode)
    torch.testing.assert_close(qr.norm(dim=-1), q.norm(dim=-1))
    torch.testing.assert_close(kr.norm(dim=-1), k.norm(dim=-1))
    torch.testing.assert_close(qr[..., 64:], q[..., 64:])
    (qr * kr).sum().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()


@pytest.mark.parametrize("mode", BASELINE_POS_MODES)
@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_pretrain_and_finetune_forward_backward(mode, use_checkpoint):
    torch.manual_seed(2)
    pt = pretrain_model(mode, use_checkpoint=use_checkpoint)
    video = torch.randn(2, 3, 4, 32, 32)
    mask = tube_mask()
    captured = {}

    def capture_decoder(module, args, kwargs):
        captured["coords"] = kwargs["rope_coords"].detach().clone()

    hook = pt.decoder.register_forward_pre_hook(capture_decoder, with_kwargs=True)
    output = pt(video, mask)
    hook.remove()
    assert output.shape == (2, 4, 1536) and torch.isfinite(output).all()
    full = build_baseline_coordinates(mode, 2, (2, 2, 2), video.device)
    expected = torch.cat((full[~mask].reshape(2, 4, -1), full[mask].reshape(2, 4, -1)), dim=1)
    torch.testing.assert_close(captured["coords"], expected)
    loss = (output - torch.randn_like(output)).square().mean()
    loss.backward()
    for parameter in [pt.encoder.patch_embed.proj.weight,
                      pt.encoder.blocks[0].attn.qkv.weight,
                      pt.decoder.blocks[0].attn.qkv.weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0

    ft = finetune_model(mode, use_checkpoint=use_checkpoint)
    encoder_state = {name[len("encoder."):]: value for name, value in pt.state_dict().items()
                     if name.startswith("encoder.")}
    loaded = ft.load_state_dict(encoder_state, strict=False)
    assert set(loaded.missing_keys) == {"fc_norm.weight", "fc_norm.bias", "head.weight", "head.bias"}
    # Pretraining's final norm is absent in the mean-pooling classifier.
    assert set(loaded.unexpected_keys) == {"norm.weight", "norm.bias"}
    logits = ft(video)
    assert logits.shape == (2, 51) and torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert ft.blocks[0].attn.qkv.weight.grad.abs().sum() > 0
    ft.eval()
    with torch.no_grad():
        assert torch.isfinite(ft(video)).all()


@pytest.mark.parametrize("mode", BASELINE_POS_MODES)
def test_checkpoint_roundtrip(mode, tmp_path):
    model = pretrain_model(mode).eval()
    filename = tmp_path / "checkpoint.pth"
    torch.save({"model": model.state_dict(), "args": {"pos_mode": mode}}, filename)
    restored = pretrain_model(mode).eval()
    restored.load_state_dict(torch.load(filename, weights_only=True)["model"], strict=True)
    video = torch.randn(2, 3, 4, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(model(video, tube_mask()), restored(video, tube_mask()))


@pytest.mark.parametrize("entrypoint", ["run_mae_pretraining.py", "run_class_finetuning.py"])
@pytest.mark.parametrize("mode", BASELINE_POS_MODES)
def test_training_cli_accepts_baseline(entrypoint, mode, monkeypatch):
    # Execute the actual parser without importing unrelated dataset libraries.
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / entrypoint).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_args")
    namespace = {"argparse": argparse, "BASELINE_POS_MODES": BASELINE_POS_MODES}
    exec(compile(ast.Module(body=[function], type_ignores=[]), entrypoint, "exec"), namespace)
    monkeypatch.setattr("sys.argv", [entrypoint, "--pos_mode", mode, "--rope_rotary_dim", "64"])
    parsed = namespace["get_args"]()
    args = parsed[0] if isinstance(parsed, tuple) else parsed
    assert args.pos_mode == mode and args.rope_rotary_dim == 64
