import torch

from stpe_rope import (
    VideoSTPE,
    apply_3d_rope,
    build_3d_coordinates,
    build_4d_coordinates,
)


def test_coordinate_order_is_t_h_w():
    temporal = torch.tensor([[0.0, 3.0]])
    coords = build_3d_coordinates(1, (2, 2, 2), temporal, temporal.device)
    expected = torch.tensor(
        [[
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 1.0, 3.0],
            [1.0, 0.0, 3.0],
            [1.0, 1.0, 3.0],
        ]]
    )
    torch.testing.assert_close(coords, expected)


def test_rope_preserves_shape_norm_and_gradient():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 8, 64, requires_grad=True)
    k = torch.randn(2, 3, 8, 64, requires_grad=True)
    temporal = torch.arange(2, dtype=torch.float32).view(1, 2).expand(2, -1)
    coords = build_3d_coordinates(2, (2, 2, 2), temporal, q.device)

    q_rot, k_rot = apply_3d_rope(q, k, coords, (20, 20, 24))
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    torch.testing.assert_close(q_rot.float().norm(dim=-1), q.float().norm(dim=-1))
    torch.testing.assert_close(k_rot.float().norm(dim=-1), k.float().norm(dim=-1))

    (q_rot.square().mean() + k_rot.square().mean()).backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()


def test_stpe_uses_only_tube_visible_tokens():
    torch.manual_seed(1)
    tokens = torch.randn(2, 4, 2, 2, 6)
    mask = torch.tensor(
        [[[[False, True], [False, True]]]], dtype=torch.bool
    ).expand(2, 4, 2, 2).clone()

    stpe = VideoSTPE(window_size=3, noise_mode="db4")
    f_reference = stpe(tokens, mask)

    changed = tokens.clone()
    changed[mask] = changed[mask] + 10000.0
    f_changed = stpe(changed, mask)

    torch.testing.assert_close(f_reference, f_changed)
    assert f_reference.shape == (2, 4)
    assert torch.isfinite(f_reference).all()
    assert torch.all(f_reference[:, 1:] >= f_reference[:, :-1])


def test_stpe_rejects_non_tube_mask():
    tokens = torch.randn(1, 4, 2, 2, 6)
    mask = torch.zeros(1, 4, 2, 2, dtype=torch.bool)
    mask[:, 0, 0, 0] = True
    stpe = VideoSTPE(window_size=3)

    try:
        stpe(tokens, mask)
    except ValueError as error:
        assert "tube masking" in str(error)
    else:
        raise AssertionError("A non-tube mask must be rejected")


def test_hwft_coordinate_order_is_t_h_w():
    observation = torch.tensor([[0.0, 3.0]])
    coords = build_4d_coordinates(
        1,
        (2, 2, 2),
        observation,
        observation.device,
    )
    expected = torch.tensor(
        [[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 1.0],
            [0.0, 1.0, 3.0, 1.0],
            [1.0, 0.0, 3.0, 1.0],
            [1.0, 1.0, 3.0, 1.0],
        ]]
    )
    torch.testing.assert_close(coords, expected)


def test_four_axis_rope_preserves_shape_norm_and_gradient():
    torch.manual_seed(2)
    q = torch.randn(2, 3, 8, 64, requires_grad=True)
    k = torch.randn(2, 3, 8, 64, requires_grad=True)
    observation = torch.tensor(
        [[0.0, 0.5], [0.0, 2.0]],
        dtype=torch.float32,
    )
    coords = build_4d_coordinates(
        2,
        (2, 2, 2),
        observation,
        q.device,
    )

    q_rot, k_rot = apply_3d_rope(
        q,
        k,
        coords,
        axis_dims=(20, 20, 12, 12),
    )

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    torch.testing.assert_close(
        q_rot.float().norm(dim=-1),
        q.float().norm(dim=-1),
    )
    torch.testing.assert_close(
        k_rot.float().norm(dim=-1),
        k.float().norm(dim=-1),
    )

    (q_rot.square().mean() + k_rot.square().mean()).backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
