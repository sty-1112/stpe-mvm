"""Parameter-free RoPE baselines for VideoMAE.

Positions are built before masking in Conv3d's T,H,W flattening order.
All baselines use one global frequency ladder and adjacent channel pairs,
matching the pairing convention of the existing HWT/HWF implementation.
"""

from typing import Sequence, Tuple
import math

import torch


BASELINE_POS_MODES = ("vanilla_rope", "tad_rope", "m_rope", "video_rope", "video_rope_f")
VIDEO_POS_MODES = ("video_rope", "video_rope_f")


def baseline_coordinate_axes(mode: str) -> int:
    if mode not in BASELINE_POS_MODES:
        raise ValueError("Unsupported baseline positional mode: {!r}".format(mode))
    return 3 if mode == "m_rope" or mode in VIDEO_POS_MODES else 1


def build_baseline_coordinates(
    mode: str,
    batch_size: int,
    grid_size: Sequence[int],
    device: torch.device,
    tad_gamma: float = 1.0,
    temporal_spacing: float = 2.0,
    temporal_coordinate: torch.Tensor = None,
) -> torch.Tensor:
    """Return [B,T*H*W,A] positions, without renumbering visible tokens."""
    baseline_coordinate_axes(mode)
    if len(grid_size) != 3 or any(int(size) <= 0 for size in grid_size):
        raise ValueError("grid_size must contain three positive sizes (T,H,W)")
    time_size, height, width = (int(size) for size in grid_size)
    if mode == "m_rope" or mode in VIDEO_POS_MODES:
        time = torch.arange(time_size, device=device, dtype=torch.float32)
        h = torch.arange(height, device=device, dtype=torch.float32)
        w = torch.arange(width, device=device, dtype=torch.float32)
        time, h, w = torch.meshgrid(time, h, w, indexing="ij")
        time, h, w = (
            value.unsqueeze(0).expand(batch_size, -1, -1, -1)
            for value in (time, h, w)
        )
        if mode == "video_rope_f":
            if temporal_coordinate is None or tuple(temporal_coordinate.shape) != (batch_size, time_size):
                raise ValueError("video_rope_f requires a temporal_coordinate tensor [B,T]")
            # Keep continuous f in float32. Never round or cast to integer IDs.
            time = temporal_coordinate.to(device=device, dtype=torch.float32)
            time = time.view(batch_size, time_size, 1, 1).expand(-1, -1, height, width)
        if mode in VIDEO_POS_MODES:
            if not math.isfinite(float(temporal_spacing)) or float(temporal_spacing) <= 0:
                raise ValueError("temporal_spacing must be finite and positive")
            time = float(temporal_spacing) * time
            # Match the official implementation's integer center offsets.
            h = time + h - (height - 1) // 2
            w = time + w - (width - 1) // 2
        # Coordinate storage is (h,w,t/f); frequency assignment depends on mode.
        return torch.stack((h, w, time), dim=-1).reshape(batch_size, -1, 3)
    positions = torch.arange(
        time_size * height * width, device=device, dtype=torch.float32
    ).view(1, -1, 1)
    if mode == "tad_rope":
        if not math.isfinite(float(tad_gamma)) or float(tad_gamma) < 0:
            raise ValueError("tad_gamma must be finite and nonnegative")
        time = torch.arange(time_size, device=device, dtype=torch.float32)
        time = time.repeat_interleave(height * width).view(1, -1, 1)
        # TC-LLaVA's dual rotation composes on the same channels:
        # R(n) R(gamma*t) = R(n + gamma*t), not a channel split.
        positions = positions + float(tad_gamma) * time
    return positions.expand(batch_size, -1, -1)


def apply_video_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    coordinates: torch.Tensor,
    mode: str,
    rotary_dim: int = 64,
    axis_dims: Sequence[int] = (24, 24, 16),
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate Q/K using a global frequency ladder; preserve any tail."""
    axes = baseline_coordinate_axes(mode)
    rotary_dim = int(rotary_dim)
    if q.ndim != 4 or q.shape != k.shape:
        raise ValueError("q/k must have matching [B,heads,N,D] shapes")
    if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > q.shape[-1]:
        raise ValueError("rotary_dim must be positive, even, and <= head dimension")
    if not math.isfinite(float(theta)) or float(theta) <= 0:
        raise ValueError("rope_theta must be finite and positive")
    if tuple(coordinates.shape) != (q.shape[0], q.shape[2], axes):
        raise ValueError("Baseline coordinates do not match Q/K token shape")

    frequencies = float(theta) ** (
        -torch.arange(0, rotary_dim, 2, device=q.device, dtype=torch.float32)
        / float(rotary_dim)
    )
    positions = coordinates.to(device=q.device, dtype=torch.float32)
    if mode == "m_rope" or mode in VIDEO_POS_MODES:
        axis_dims = tuple(int(dim) for dim in axis_dims)
        if len(axis_dims) != 3 or any(dim <= 0 or dim % 2 for dim in axis_dims):
            raise ValueError("baseline axis_dims must be three positive even integers (h,w,t)")
        if sum(axis_dims) != rotary_dim:
            raise ValueError("sum(axis_dims) must equal rope_rotary_dim for 3D baselines")
        h_pairs, w_pairs, t_pairs = (dim // 2 for dim in axis_dims)
        if mode == "m_rope":
            pair_axes = [2] * t_pairs + [0] * h_pairs + [1] * w_pairs
        else:
            if h_pairs != w_pairs:
                raise ValueError("VideoRoPE requires equal h/w dimensions for spatial interleaving")
            pair_axes = [0, 1] * h_pairs + [2] * t_pairs
        angles = positions[..., pair_axes] * frequencies
    else:
        angles = positions[..., 0:1] * frequencies
    cos = angles.cos().unsqueeze(1)
    sin = angles.sin().unsqueeze(1)

    def rotate(x):
        work = x[..., :rotary_dim].float()
        even, odd = work[..., 0::2], work[..., 1::2]
        rotated = torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        ).flatten(-2).to(x.dtype)
        return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)

    return rotate(q), rotate(k)
