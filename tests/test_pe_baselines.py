"""CPU smoke tests of real VideoMAE encoder/decoder/classifier paths."""

import argparse
import ast
from pathlib import Path

import pytest
import torch
from timm.models import create_model

from modeling_finetune import VisionTransformer
from modeling_pretrain import PretrainVisionTransformer
from video_rope import BASELINE_POS_MODES, apply_video_rope, build_baseline_coordinates as _build_coordinates
from stpe_rope import VideoSTPE
from temporal_pe_v2 import VideoSTPEV2


torch.set_num_threads(1)


def build_baseline_coordinates(mode, batch_size, grid_size, device, **kwargs):
    # Coordinate-only generic checks supply f=t; estimator behavior is tested below.
    if mode == "video_rope_f" and "temporal_coordinate" not in kwargs:
        kwargs["temporal_coordinate"] = torch.arange(grid_size[0], device=device).float().expand(batch_size, -1)
    return _build_coordinates(mode, batch_size, grid_size, device, **kwargs)


def pretrain_model(mode, **kwargs):
    # Keep ViT-S encoder/decoder head dimensions (64) with a small smoke grid.
    return PretrainVisionTransformer(
        img_size=32, num_frames=kwargs.pop("num_frames", 4), patch_size=16, tubelet_size=2,
        encoder_embed_dim=384, encoder_num_heads=6, encoder_depth=2,
        decoder_embed_dim=192, decoder_num_heads=3, decoder_depth=1,
        qkv_bias=True, pos_mode=mode, rope_axis_dims=(24, 24, 16),
        **kwargs,
    )


def finetune_model(mode, **kwargs):
    return VisionTransformer(
        img_size=32, all_frames=kwargs.pop("all_frames", 4), patch_size=16, tubelet_size=2,
        embed_dim=384, num_heads=6, depth=2, num_classes=51,
        qkv_bias=True, pos_mode=mode, rope_axis_dims=(24, 24, 16),
        init_scale=1.0, **kwargs,
    )


def tube_mask(time_size=2):
    return torch.tensor([[False, True, False, True] * time_size] * 2)


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


def test_tad_dual_rotation_and_gamma_zero():
    q = torch.randn(2, 3, 8, 64)
    vanilla = build_baseline_coordinates("vanilla_rope", 2, (2, 2, 2), q.device)
    tad = build_baseline_coordinates("tad_rope", 2, (2, 2, 2), q.device, tad_gamma=3.)
    torch.testing.assert_close(tad[0, :, 0], torch.tensor([0., 1., 2., 3., 7., 8., 9., 10.]))
    qr, _ = apply_video_rope(q, q, vanilla, "vanilla_rope")
    time = tad - vanilla
    composed, _ = apply_video_rope(qr, qr, time, "vanilla_rope")
    actual, _ = apply_video_rope(q, q, tad, "tad_rope")
    torch.testing.assert_close(actual, composed)
    zero = build_baseline_coordinates("tad_rope", 2, (2, 2, 2), q.device, tad_gamma=0.)
    torch.testing.assert_close(zero, vanilla)


def test_tad_gamma_reaches_model_coordinate_builder():
    model = pretrain_model("tad_rope", tad_gamma=4.)
    _, coords = model.encoder.forward_features(torch.randn(2, 3, 4, 32, 32), tube_mask())
    expected = build_baseline_coordinates("tad_rope", 2, (2, 2, 2), torch.device("cpu"), tad_gamma=4.)
    torch.testing.assert_close(coords, expected)


def test_mrope_global_frequency_assignment():
    q = torch.randn(2, 3, 8, 64)
    coords = build_baseline_coordinates("m_rope", 2, (2, 2, 2), q.device)
    torch.testing.assert_close(coords[0, 4], torch.tensor([0., 0., 1.]))
    torch.testing.assert_close(coords[0, 3], torch.tensor([1., 1., 0.]))
    actual, _ = apply_video_rope(q, q, coords, "m_rope")
    expected = q.clone()
    axes = [2] * 8 + [0] * 12 + [1] * 12
    for pair, axis in enumerate(axes):
        angle = coords[..., axis].unsqueeze(1) / (10000.0 ** (2 * pair / 64))
        a, b = q[..., 2 * pair], q[..., 2 * pair + 1]
        expected[..., 2 * pair] = a * angle.cos() - b * angle.sin()
        expected[..., 2 * pair + 1] = a * angle.sin() + b * angle.cos()
    torch.testing.assert_close(actual, expected)


def test_mrope_equal_axis_positions_reduce_to_vanilla():
    q = torch.randn(2, 3, 8, 64)
    one = build_baseline_coordinates("vanilla_rope", 2, (2, 2, 2), q.device)
    three = one.expand(-1, -1, 3)
    vanilla, _ = apply_video_rope(q, q, one, "vanilla_rope")
    mrope, _ = apply_video_rope(q, q, three, "m_rope")
    torch.testing.assert_close(mrope, vanilla)


def test_videorope_diagonal_layout_and_spacing():
    coords = build_baseline_coordinates("video_rope", 1, (2, 3, 3), torch.device("cpu"), temporal_spacing=1.5)
    torch.testing.assert_close(coords[0, 4], torch.tensor([0., 0., 0.]))
    torch.testing.assert_close(coords[0, 13], torch.tensor([1.5, 1.5, 1.5]))
    torch.testing.assert_close(coords[0, 9] - coords[0, 0], torch.tensor([1.5, 1.5, 1.5]))
    torch.testing.assert_close(coords[0, 0], torch.tensor([-1., -1., 0.]))


def test_videorope_interleaved_spatial_and_low_frequency_time_reference():
    q = torch.randn(2, 3, 8, 64)
    coords = build_baseline_coordinates("video_rope", 2, (2, 2, 2), q.device)
    actual, _ = apply_video_rope(q, q, coords, "video_rope")
    expected = q.clone()
    axes = [0, 1] * 12 + [2] * 8
    for pair, axis in enumerate(axes):
        angle = coords[..., axis].unsqueeze(1) / (10000.0 ** (2 * pair / 64))
        a, b = q[..., 2 * pair], q[..., 2 * pair + 1]
        expected[..., 2 * pair] = a * angle.cos() - b * angle.sin()
        expected[..., 2 * pair + 1] = a * angle.sin() + b * angle.cos()
    torch.testing.assert_close(actual, expected)


def test_videorope_spacing_reaches_encoder_and_classifier():
    video = torch.randn(2, 3, 4, 32, 32)
    pt = pretrain_model("video_rope", temporal_spacing=3.5)
    _, coords = pt.encoder.forward_features(video, tube_mask())
    expected = build_baseline_coordinates("video_rope", 2, (2, 2, 2), video.device, temporal_spacing=3.5)
    torch.testing.assert_close(coords, expected)
    ft = finetune_model("video_rope", temporal_spacing=3.5)
    captured = {}
    def capture(module, args):
        captured["coords"] = args[1]
    hook = ft.blocks[0].register_forward_pre_hook(capture)
    ft(video)
    hook.remove()
    torch.testing.assert_close(captured["coords"], expected)


@pytest.mark.parametrize("spacing", [0., -1., float("nan"), float("inf")])
def test_videorope_rejects_invalid_spacing(spacing):
    with pytest.raises(ValueError):
        build_baseline_coordinates("video_rope", 1, (2, 2, 2), torch.device("cpu"), temporal_spacing=spacing)


def test_videorope_rejects_unequal_spatial_channel_budget():
    q = torch.randn(1, 1, 8, 64)
    coords = build_baseline_coordinates("video_rope", 1, (2, 2, 2), q.device)
    with pytest.raises(ValueError, match="equal h/w"):
        apply_video_rope(q, q, coords, "video_rope", axis_dims=(20, 28, 16))


def test_video_f_preserves_fractional_positions_in_all_axes():
    f = torch.tensor([[0., .25, 2.75, 3.], [0., 1., 1.5, 3.]])
    coords = build_baseline_coordinates("video_rope_f", 2, (4, 2, 2), f.device, temporal_coordinate=f)
    torch.testing.assert_close(coords[0, 4] - coords[0, 0], torch.tensor([.5, .5, .5]))
    torch.testing.assert_close(coords[0, 8] - coords[0, 0], torch.tensor([5.5, 5.5, 5.5]))
    torch.testing.assert_close(coords[1, 4] - coords[1, 0], torch.tensor([2., 2., 2.]))
    assert coords.dtype == torch.float32


def test_video_f_equals_video_t_when_f_is_raw_time():
    q = torch.randn(2, 3, 16, 64)
    t = build_baseline_coordinates("video_rope", 2, (4, 2, 2), q.device)
    f = build_baseline_coordinates("video_rope_f", 2, (4, 2, 2), q.device)
    torch.testing.assert_close(t, f)
    qr_t, _ = apply_video_rope(q, q, t, "video_rope")
    qr_f, _ = apply_video_rope(q, q, f, "video_rope_f")
    torch.testing.assert_close(qr_t, qr_f)


def test_video_f_beta_zero_recovers_pretraining_and_classification():
    video = torch.randn(2, 3, 8, 32, 32)
    pt_t = pretrain_model("video_rope", num_frames=8).eval()
    pt_f = pretrain_model("video_rope_f", num_frames=8, stpe_mix_beta=0.).eval()
    pt_f.load_state_dict(pt_t.state_dict(), strict=True)
    ft_t = finetune_model("video_rope", all_frames=8).eval()
    ft_f = finetune_model("video_rope_f", all_frames=8, stpe_mix_beta=0.).eval()
    ft_f.load_state_dict(ft_t.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(pt_t(video, tube_mask(4)), pt_f(video, tube_mask(4)))
        torch.testing.assert_close(ft_t(video), ft_f(video))


def test_video_f_uses_visible_features_once_and_reuses_decoder_coordinate():
    pt = pretrain_model("video_rope_f", num_frames=8, stpe_window_size=1, stpe_noise_mode="none")
    video = torch.randn(2, 3, 8, 32, 32)
    video[:, :, 4:6] *= 3.
    video[:, :, 6:8] *= 8.
    mask = tube_mask(4)
    observed_f = []
    decoder_coordinates = []
    hook_f = pt.encoder.stpe.register_forward_hook(lambda module, args, output: observed_f.append(output.clone()))
    hook_d = pt.decoder.register_forward_pre_hook(
        lambda module, args, kwargs: decoder_coordinates.append(kwargs["rope_coords"].clone()), with_kwargs=True
    )
    output = pt(video, mask)
    hook_f.remove()
    hook_d.remove()
    assert len(observed_f) == 1 and len(decoder_coordinates) == 1
    f = observed_f[0]
    torch.testing.assert_close(f[:, 0], torch.zeros(2))
    torch.testing.assert_close(f[:, -1], torch.full((2,), 3.))
    assert torch.all(f[:, 1:] >= f[:, :-1])
    assert not torch.allclose(f, torch.arange(4).float().expand(2, -1))
    full = build_baseline_coordinates("video_rope_f", 2, (4, 2, 2), video.device, temporal_coordinate=f)
    expected = torch.cat((full[~mask].reshape(2, 8, 3), full[mask].reshape(2, 8, 3)), dim=1)
    torch.testing.assert_close(decoder_coordinates[0], expected)
    # Masked tubelets occupy the right half of each frame; Conv3d patches do not overlap.
    changed = video.clone()
    changed[:, :, :, :, 16:] += 10000.
    with torch.no_grad():
        ref_vis, ref_coords = pt.encoder.forward_features(video, mask)
        changed_vis, changed_coords = pt.encoder.forward_features(changed, mask)
    torch.testing.assert_close(ref_coords, changed_coords)
    torch.testing.assert_close(ref_vis, changed_vis)
    output.square().mean().backward()
    assert torch.isfinite(pt.encoder.patch_embed.proj.weight.grad).all()


def test_video_f_estimator_selection_and_legacy_forward_backward():
    legacy = pretrain_model("video_rope_f", stpe_estimator="v1", stpe_noise_mode="none")
    assert type(legacy.encoder.stpe) is VideoSTPE
    assert isinstance(pretrain_model("video_rope_f").encoder.stpe, VideoSTPEV2)
    assert type(finetune_model("video_rope_f", stpe_estimator="v1").stpe) is VideoSTPE
    output = legacy(torch.randn(2, 3, 4, 32, 32), tube_mask())
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(legacy.decoder.blocks[0].attn.qkv.weight.grad).all()


def test_video_f_rejects_non_tube_mask():
    mask = tube_mask()
    mask[0, 0], mask[0, 1] = True, False
    with pytest.raises(ValueError, match="tube masking"):
        pretrain_model("video_rope_f")(torch.randn(2, 3, 4, 32, 32), mask)


@pytest.mark.parametrize("coordinate", [None, torch.zeros(1, 3)])
def test_video_f_requires_matching_temporal_shape(coordinate):
    with pytest.raises(ValueError, match="temporal_coordinate"):
        _build_coordinates("video_rope_f", 2, (2, 2, 2), torch.device("cpu"), temporal_coordinate=coordinate)


@pytest.mark.parametrize("mode", BASELINE_POS_MODES)
def test_cpu_bfloat16_autocast_forward_backward(mode):
    pt = pretrain_model(mode)
    ft = finetune_model(mode)
    video = torch.randn(2, 3, 4, 32, 32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        reconstruction = pt(video, tube_mask())
        logits = ft(video)
        loss = reconstruction.float().square().mean() + torch.nn.functional.cross_entropy(logits.float(), torch.tensor([0, 1]))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pt.decoder.blocks[0].attn.qkv.weight.grad).all()
    assert torch.isfinite(ft.blocks[0].attn.qkv.weight.grad).all()


@pytest.mark.parametrize("dims", [(12, 12, 8), (23, 25, 16), (32, 32)])
def test_mrope_rejects_invalid_dimension_budget(dims):
    q = torch.randn(1, 1, 8, 64)
    coords = build_baseline_coordinates("m_rope", 1, (2, 2, 2), q.device)
    with pytest.raises(ValueError):
        apply_video_rope(q, q, coords, "m_rope", axis_dims=dims)


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
    classifier = finetune_model(mode).eval()
    torch.save({"model": classifier.state_dict()}, filename)
    reloaded_classifier = finetune_model(mode).eval()
    reloaded_classifier.load_state_dict(torch.load(filename, weights_only=True)["model"], strict=True)
    with torch.no_grad():
        torch.testing.assert_close(classifier(video), reloaded_classifier(video))


@pytest.mark.parametrize("mode", BASELINE_POS_MODES)
def test_registered_vit_small_factories_accept_training_options(mode):
    options = dict(pos_mode=mode, rope_rotary_dim=64, rope_axis_dims=(24, 24, 16),
                   tad_gamma=1., temporal_spacing=2., stpe_estimator="v2")
    pt = create_model("pretrain_videomae_small_patch16_224", pretrained=False,
                      drop_block_rate=None, decoder_depth=4, num_frames=16, **options)
    assert len(pt.encoder.blocks) == 12 and len(pt.decoder.blocks) == 4
    assert pt.encoder.patch_embed.grid_size == (8, 14, 14)
    assert pt.encoder.blocks[0].attn.rope_mode == mode
    assert pt.decoder.blocks[0].attn.rope_mode == mode
    ft = create_model("vit_small_patch16_224", pretrained=False,
                      drop_block_rate=None, all_frames=16, num_classes=51, **options)
    assert len(ft.blocks) == 12 and ft.head.out_features == 51
    assert ft.blocks[0].attn.rope_mode == mode


@pytest.mark.parametrize("entrypoint", ["run_mae_pretraining.py", "run_class_finetuning.py"])
@pytest.mark.parametrize("mode", BASELINE_POS_MODES)
def test_training_cli_accepts_baseline(entrypoint, mode, monkeypatch):
    # Execute the actual parser without importing unrelated dataset libraries.
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / entrypoint).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_args")
    namespace = {"argparse": argparse, "BASELINE_POS_MODES": BASELINE_POS_MODES}
    exec(compile(ast.Module(body=[function], type_ignores=[]), entrypoint, "exec"), namespace)
    monkeypatch.setattr("sys.argv", [entrypoint, "--pos_mode", mode, "--rope_rotary_dim", "64", "--tad_gamma", "3", "--temporal_spacing", "1.5", "--stpe_estimator", "v1"])
    parsed = namespace["get_args"]()
    args = parsed[0] if isinstance(parsed, tuple) else parsed
    assert args.pos_mode == mode and args.rope_rotary_dim == 64
    assert args.tad_gamma == 3.
    assert args.temporal_spacing == 1.5
    assert args.stpe_estimator == "v1"
