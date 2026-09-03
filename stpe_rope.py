"""Spatiotemporal coordinates and 3D RoPE for VideoMAE.

The module is intentionally parameter-free.  Fixed tensors are registered as
non-persistent buffers, so enabling RoPE/STPE does not change checkpoint keys.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn


VALID_POS_MODES = ('original', 'hwt_rope', 'hwf_rope', 'hwft_rope', 'hwf_v2_rope')
VALID_FINETUNE_POS_MODES = VALID_POS_MODES


def validate_pos_mode(pos_mode: str) -> str:
    if pos_mode not in VALID_POS_MODES:
        raise ValueError(
            "Unsupported pos_mode={!r}; expected one of {}".format(
                pos_mode, VALID_POS_MODES
            )
        )
    return pos_mode


def validate_finetune_pos_mode(pos_mode: str) -> str:
    if pos_mode not in VALID_FINETUNE_POS_MODES:
        raise ValueError(
            "Unsupported finetune pos_mode={!r}; expected one of {}".format(
                pos_mode, VALID_FINETUNE_POS_MODES
            )
        )
    return pos_mode


def build_3d_coordinates(
    batch_size: int,
    grid_size: Sequence[int],
    temporal_coordinate: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build coordinates in the same flattened order as Conv3d: T, H, W.

    Args:
        batch_size: Batch size B.
        grid_size: Tuple/list (T, H, W).
        temporal_coordinate: Raw t or STPE f with shape [B, T].
        device: Output device.

    Returns:
        Float32 coordinates [B, T*H*W, 3] ordered as (h, w, t_or_f).
    """
    if len(grid_size) != 3:
        raise ValueError("grid_size must be (T, H, W), got {}".format(grid_size))
    t_size, h_size, w_size = [int(value) for value in grid_size]
    expected_temporal_shape = (batch_size, t_size)
    if tuple(temporal_coordinate.shape) != expected_temporal_shape:
        raise ValueError(
            "temporal_coordinate must have shape {}, got {}".format(
                expected_temporal_shape, tuple(temporal_coordinate.shape)
            )
        )

    temporal_coordinate = temporal_coordinate.to(device=device, dtype=torch.float32)
    h_coord = torch.arange(h_size, device=device, dtype=torch.float32)
    w_coord = torch.arange(w_size, device=device, dtype=torch.float32)

    h_coord = h_coord.view(1, 1, h_size, 1).expand(
        batch_size, t_size, h_size, w_size
    )
    w_coord = w_coord.view(1, 1, 1, w_size).expand(
        batch_size, t_size, h_size, w_size
    )
    temporal_coordinate = temporal_coordinate.view(
        batch_size, t_size, 1, 1
    ).expand(batch_size, t_size, h_size, w_size)

    coordinates = torch.stack(
        (h_coord, w_coord, temporal_coordinate), dim=-1
    )
    return coordinates.reshape(batch_size, t_size * h_size * w_size, 3)


def build_4d_coordinates(
    batch_size: int,
    grid_size: Sequence[int],
    observation_coordinate: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build HWFT coordinates ordered as (h, w, f, t).

    The token flattening order remains T, H, W.  The f coordinate is produced
    by VideoSTPE, while t is the original tubelet index 0..T-1.

    Returns:
        Float32 coordinates [B, T*H*W, 4] ordered as (h, w, f, t).
    """
    hwf_coordinates = build_3d_coordinates(
        batch_size,
        grid_size,
        observation_coordinate,
        device,
    )

    t_size, h_size, w_size = [int(value) for value in grid_size]
    time_coordinate = torch.arange(
        t_size,
        device=device,
        dtype=torch.float32,
    )
    time_coordinate = time_coordinate.view(
        1, t_size, 1, 1
    ).expand(
        batch_size, t_size, h_size, w_size
    )
    time_coordinate = time_coordinate.reshape(
        batch_size,
        t_size * h_size * w_size,
        1,
    )

    return torch.cat((hwf_coordinates, time_coordinate), dim=-1)


def _apply_1d_rope(
    x: torch.Tensor,
    coordinate: torch.Tensor,
    theta: float,
) -> torch.Tensor:
    """Apply one-axis RoPE to [B, heads, tokens, axis_dim]."""
    axis_dim = x.shape[-1]
    if axis_dim % 2 != 0:
        raise ValueError("Every RoPE axis dimension must be even, got {}".format(axis_dim))
    if coordinate.ndim != 2:
        raise ValueError(
            "coordinate must have shape [B, N], got {}".format(
                tuple(coordinate.shape)
            )
        )
    if coordinate.shape[0] != x.shape[0] or coordinate.shape[1] != x.shape[-2]:
        raise ValueError(
            "Coordinate/token mismatch: x={}, coordinate={}".format(
                tuple(x.shape), tuple(coordinate.shape)
            )
        )

    work = x.float()
    exponent = torch.arange(
        0, axis_dim, 2, device=x.device, dtype=torch.float32
    ) / float(axis_dim)
    inv_freq = torch.pow(
        torch.tensor(float(theta), device=x.device, dtype=torch.float32),
        -exponent,
    )
    angle = coordinate.to(device=x.device, dtype=torch.float32).unsqueeze(-1)
    angle = angle * inv_freq.view(1, 1, -1)
    cos = angle.cos().unsqueeze(1)
    sin = angle.sin().unsqueeze(1)

    even = work[..., 0::2]
    odd = work[..., 1::2]
    rotated = torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos), dim=-1
    ).flatten(-2)
    return rotated.to(dtype=x.dtype)


def apply_3d_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    coordinates: torch.Tensor,
    axis_dims: Sequence[int] = (20, 20, 24),
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply independent RoPE rotations for every coordinate axis to Q and K.

    Any head dimensions after sum(axis_dims) are preserved.  This keeps the
    original model runnable for backbones whose head dimension is larger than
    ViT-S/ViT-B's 64.
    """
    if q.shape != k.shape:
        raise ValueError("q and k must have identical shapes")
    if q.ndim != 4:
        raise ValueError("q/k must have shape [B, heads, N, D]")
    if coordinates.ndim != 3 or coordinates.shape[-1] not in (3, 4):
        raise ValueError(
            "coordinates must have shape [B, N, 3] or [B, N, 4]"
        )

    num_axes = int(coordinates.shape[-1])
    axis_dims = tuple(int(dim) for dim in axis_dims)
    if len(axis_dims) != num_axes:
        raise ValueError(
            "axis_dims must contain exactly {} integers for coordinates "
            "with {} axes, got {}".format(
                num_axes, num_axes, len(axis_dims)
            )
        )
    if any(dim <= 0 or dim % 2 != 0 for dim in axis_dims):
        raise ValueError("axis_dims must be positive even integers")
    rotary_dim = sum(axis_dims)
    if rotary_dim > q.shape[-1]:
        raise ValueError(
            "sum(axis_dims)={} exceeds attention head dimension={}".format(
                rotary_dim, q.shape[-1]
            )
        )
    if coordinates.shape[0] != q.shape[0] or coordinates.shape[1] != q.shape[-2]:
        raise ValueError(
            "coordinates {} do not match q/k token shape {}".format(
                tuple(coordinates.shape), tuple(q.shape)
            )
        )

    q_parts = []
    k_parts = []
    start = 0
    for axis, axis_dim in enumerate(axis_dims):
        end = start + axis_dim
        q_parts.append(
            _apply_1d_rope(q[..., start:end], coordinates[..., axis], theta)
        )
        k_parts.append(
            _apply_1d_rope(k[..., start:end], coordinates[..., axis], theta)
        )
        start = end

    if start < q.shape[-1]:
        q_parts.append(q[..., start:])
        k_parts.append(k[..., start:])
    return torch.cat(q_parts, dim=-1), torch.cat(k_parts, dim=-1)


class VideoSTPE(nn.Module):
    """Compute a per-tubelet observation-difference coordinate f.

    The computation uses unpositioned PatchEmbed features.  During MVM
    pretraining, only tube-mask-visible features are read.  The implementation
    follows the supplied STPE core: local E[v^2], wavelet/MAD noise correction,
    per-sequence scale calibration, and cumulative delta-f.
    """

    # PyWavelets db4 decomposition high-pass filter coefficients.
    _DB4_HIGH_PASS = (
        -0.23037781330885523,
        0.7148465705525415,
        -0.6308807679295904,
        -0.02798376941698385,
        0.18703481171888114,
        0.030841381835986965,
        -0.032883011666982945,
        -0.010597401784997278,
    )

    def __init__(
        self,
        window_size: int = 5,
        noise_mode: str = "db4",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if int(window_size) < 1:
            raise ValueError("window_size must be >= 1")
        if noise_mode not in ("db4", "none"):
            raise ValueError("noise_mode must be 'db4' or 'none'")
        self.window_size = int(window_size)
        self.noise_mode = noise_mode
        self.eps = float(eps)
        self.register_buffer(
            "db4_high_pass",
            torch.tensor(self._DB4_HIGH_PASS, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _window_bounds(length: int, window_size: int, index: int) -> Tuple[int, int]:
        half_left = window_size // 2
        half_right = window_size - 1 - half_left
        left = max(0, index - half_left)
        right = min(length, index + half_right + 1)
        return left, right

    def _window_mean_square(self, variation: torch.Tensor) -> torch.Tensor:
        values = variation.square()
        outputs = []
        for index in range(values.shape[1]):
            left, right = self._window_bounds(
                values.shape[1], self.window_size, index
            )
            outputs.append(values[:, left:right].mean(dim=1))
        return torch.stack(outputs, dim=1)

    def _db4_detail(self, signal: torch.Tensor) -> torch.Tensor:
        """Stationary high-pass filtering with circular boundaries.

        Signal length is kept unchanged.  The fixed phase convention has no
        trainable state and is consistent across all experiments.
        """
        batch_size, length = signal.shape
        filter_size = self.db4_high_pass.numel()
        offsets = torch.arange(filter_size, device=signal.device)
        offsets = offsets - filter_size // 2
        centers = torch.arange(length, device=signal.device).unsqueeze(1)
        indices = torch.remainder(centers + offsets.unsqueeze(0), length).long()
        windows = signal[:, indices]
        filt = self.db4_high_pass.to(device=signal.device, dtype=signal.dtype)
        return (windows * filt.view(1, 1, filter_size)).sum(dim=-1)

    def _estimate_noise(self, observed_signal: torch.Tensor) -> torch.Tensor:
        if self.noise_mode == "none" or observed_signal.shape[1] < 2:
            return torch.zeros_like(observed_signal)
        high = self._db4_detail(observed_signal)
        estimates = []
        for index in range(high.shape[1]):
            left, right = self._window_bounds(
                high.shape[1], self.window_size, index
            )
            local = high[:, left:right]
            median = local.median(dim=1, keepdim=True).values
            mad = (local - median).abs().median(dim=1).values
            estimates.append(mad.square())
        return torch.stack(estimates, dim=1)

    def _select_visible_tokens(
        self,
        patch_tokens: torch.Tensor,
        masked_pos: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, time_size, height, width, channels = patch_tokens.shape
        flat_tokens = patch_tokens.reshape(
            batch_size, time_size, height * width, channels
        )
        if masked_pos is None:
            return flat_tokens
        if tuple(masked_pos.shape) != (batch_size, time_size, height, width):
            raise ValueError(
                "masked_pos must have shape {}, got {}".format(
                    (batch_size, time_size, height, width),
                    tuple(masked_pos.shape),
                )
            )
        masked_pos = masked_pos.to(device=patch_tokens.device, dtype=torch.bool)
        reference = masked_pos[:, :1].expand_as(masked_pos)
        if not torch.equal(masked_pos, reference):
            raise ValueError(
                "hwf_rope pretraining requires tube masking so visible spatial "
                "tokens are aligned across time"
            )
        visible = ~masked_pos.reshape(batch_size, time_size, height * width)
        visible_count = visible.sum(dim=-1)
        if torch.any(visible_count == 0):
            raise ValueError("Every temporal step must contain visible tokens")
        num_visible = int(visible_count[0, 0].item())
        if not torch.all(visible_count == num_visible):
            raise ValueError("Every temporal step must have the same visible count")
        return flat_tokens[visible].reshape(
            batch_size, time_size, num_visible, channels
        )

    def _write_diagnostics(
        self,
        *,
        observed_signal: torch.Tensor,
        variation: torch.Tensor,
        noise: torch.Tensor,
        expected_v2: torch.Tensor,
        gamma: torch.Tensor,
        gamma_star: torch.Tensor,
        scale: torch.Tensor,
        median_dt: torch.Tensor,
        alpha: torch.Tensor,
        temporal_coordinate: torch.Tensor,
    ) -> None:
        """Optionally append raw STPE statistics to a rank-local JSONL file.

        Diagnostics are disabled unless STPE_STATS_DIR is set. The first alpha
        value is retained in the raw record, but downstream summaries exclude
        alpha[:, 0] because it does not contribute to delta_f.
        """
        stats_dir = os.environ.get("STPE_STATS_DIR", "").strip()

        if not stats_dir:
            return

        os.makedirs(stats_dir, exist_ok=True)

        rank = os.environ.get(
            "RANK",
            os.environ.get("SLURM_PROCID", "0"),
        )

        if not hasattr(self, "_stpe_stats_step"):
            self._stpe_stats_step = 0

        self._stpe_stats_step += 1

        output_file = os.path.join(
            stats_dir,
            "stpe_stats_rank{}.jsonl".format(rank),
        )

        denominator = (
            scale * median_dt + self.eps
        ).detach().float()

        eps_fraction = (
            self.eps / denominator
        ).detach().float()

        def finite_list(value):
            value = value.detach().float()
            value = torch.nan_to_num(
                value,
                nan=0.0,
                posinf=1.0e30,
                neginf=-1.0e30,
            )
            return value.cpu().tolist()

        def finite_scalar(value):
            value = value.detach().float()
            value = torch.nan_to_num(
                value,
                nan=0.0,
                posinf=1.0e30,
                neginf=-1.0e30,
            )
            return float(value.item())

        with open(
            output_file,
            mode="a",
            encoding="utf-8",
        ) as handle:
            for sample_index in range(alpha.shape[0]):
                raw_alpha = alpha[sample_index].detach().float()
                raw_f = (
                    temporal_coordinate[sample_index]
                    .detach()
                    .float()
                )

                record = {
                    "rank": int(rank),
                    "step": int(self._stpe_stats_step),
                    "sample_in_batch": int(sample_index),
                    "window_size": int(self.window_size),
                    "noise_mode": self.noise_mode,
                    "eps": float(self.eps),
                    "time_size": int(alpha.shape[1]),
                    "observed_signal": finite_list(
                        observed_signal[sample_index]
                    ),
                    "variation": finite_list(
                        variation[sample_index]
                    ),
                    "noise": finite_list(
                        noise[sample_index]
                    ),
                    "expected_v2": finite_list(
                        expected_v2[sample_index]
                    ),
                    "gamma": finite_list(
                        gamma[sample_index]
                    ),
                    "gamma_star": finite_scalar(
                        gamma_star[sample_index]
                    ),
                    "scale": finite_scalar(
                        scale[sample_index]
                    ),
                    "median_dt": finite_scalar(
                        median_dt[sample_index]
                    ),
                    "denominator": finite_scalar(
                        denominator[sample_index]
                    ),
                    "eps_fraction": finite_scalar(
                        eps_fraction[sample_index]
                    ),
                    "alpha": finite_list(raw_alpha),
                    "f": finite_list(raw_f),
                    "alpha_nonfinite_count": int(
                        (~torch.isfinite(raw_alpha)).sum().item()
                    ),
                    "f_nonfinite_count": int(
                        (~torch.isfinite(raw_f)).sum().item()
                    ),
                }

                handle.write(
                    json.dumps(
                        record,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )

    @torch.no_grad()
    def forward(
        self,
        patch_tokens: torch.Tensor,
        masked_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if patch_tokens.ndim != 5:
            raise ValueError(
                "patch_tokens must have shape [B, T, H, W, C], got {}".format(
                    tuple(patch_tokens.shape)
                )
            )
        visible_tokens = self._select_visible_tokens(
            patch_tokens.detach(), masked_pos
        ).float()
        batch_size, time_size = visible_tokens.shape[:2]
        if time_size < 2:
            return torch.zeros(
                batch_size,
                time_size,
                device=patch_tokens.device,
                dtype=torch.float32,
            )

        difference = visible_tokens[:, 1:] - visible_tokens[:, :-1]
        variation_tail = difference.square().mean(dim=(-1, -2)).clamp_min(0).sqrt()
        variation = torch.cat((variation_tail[:, :1], variation_tail), dim=1)

        # A scalar observation with the same feature units is used for the
        # wavelet/MAD noise estimate.
        observed_signal = visible_tokens.square().mean(dim=(-1, -2)).clamp_min(0).sqrt()
        noise = self._estimate_noise(observed_signal)
        expected_v2 = self._window_mean_square(variation)
        gamma = expected_v2 - 2.0 * noise

        # With tubelet indices 0..T-1, all delta-t values are one.  Keep the
        # explicit tensor to preserve the general STPE construction.
        time_index = torch.arange(
            time_size, device=patch_tokens.device, dtype=torch.float32
        ).view(1, time_size).expand(batch_size, -1)
        dt_steps = (time_index[:, 1:] - time_index[:, :-1]).clamp_min(self.eps)
        median_dt = dt_steps.median(dim=1).values.clamp_min(self.eps)
        gamma_star = gamma.median(dim=1).values
        scale = (gamma_star / median_dt).clamp_min(self.eps)

        alpha = torch.relu(
            gamma / (scale.unsqueeze(1) * median_dt.unsqueeze(1) + self.eps)
        )
        delta_f = torch.zeros_like(alpha)
        delta_f[:, 1:] = dt_steps * alpha[:, 1:]
        temporal_coordinate = torch.cumsum(delta_f, dim=1)
        temporal_coordinate = torch.nan_to_num(
            temporal_coordinate,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        self._write_diagnostics(
            observed_signal=observed_signal,
            variation=variation,
            noise=noise,
            expected_v2=expected_v2,
            gamma=gamma,
            gamma_star=gamma_star,
            scale=scale,
            median_dt=median_dt,
            alpha=alpha,
            temporal_coordinate=temporal_coordinate,
        )

        return temporal_coordinate
