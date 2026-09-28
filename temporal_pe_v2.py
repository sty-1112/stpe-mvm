"""Stable observation-difference temporal coordinate for VideoMAE.

V2 keeps the original gamma estimator but calibrates its scale so that the
total observation-difference coordinate span equals the raw temporal span.
The module is parameter-free.
"""

from __future__ import annotations

import torch

from stpe_rope import VideoSTPE


class VideoSTPEV2(VideoSTPE):
    """Compute f using positive-gamma total-span-preserving calibration."""

    def __init__(
        self,
        window_size=5,
        noise_mode="db4",
        eps=1e-6,
        mix_beta=1.0,
    ):
        super().__init__(
            window_size=window_size,
            noise_mode=noise_mode,
            eps=eps,
        )
        mix_beta = float(mix_beta)
        if not 0.0 <= mix_beta <= 1.0:
            raise ValueError(
                "mix_beta must be in [0, 1], got {}".format(mix_beta)
            )
        self.mix_beta = mix_beta

    @torch.no_grad()
    def forward(self, patch_tokens, masked_pos=None):
        if patch_tokens.ndim != 5:
            raise ValueError(
                "patch_tokens must have shape [B, T, H, W, C], got {}".format(
                    tuple(patch_tokens.shape)
                )
            )

        # During pretraining, the inherited helper reads only visible
        # tube-mask-aligned PatchEmbed tokens.
        visible_tokens = self._select_visible_tokens(
            patch_tokens.detach(),
            masked_pos,
        ).float()

        batch_size, time_size = visible_tokens.shape[:2]

        if time_size < 2:
            return torch.zeros(
                batch_size,
                time_size,
                device=patch_tokens.device,
                dtype=torch.float32,
            )

        # Observation difference and local observation variation.
        difference = visible_tokens[:, 1:] - visible_tokens[:, :-1]

        variation_tail = (
            difference.square()
            .mean(dim=(-1, -2))
            .clamp_min(0)
            .sqrt()
        )

        # Preserve the original length-T statistic construction.
        variation = torch.cat(
            (variation_tail[:, :1], variation_tail),
            dim=1,
        )

        observed_signal = (
            visible_tokens.square()
            .mean(dim=(-1, -2))
            .clamp_min(0)
            .sqrt()
        )

        # Keep the original noise estimator and gamma definition:
        #
        # gamma_k = E[v_k^2] - 2 r_k
        noise = self._estimate_noise(observed_signal)
        expected_v2 = self._window_mean_square(variation)
        gamma = expected_v2 - 2.0 * noise

        time_index = torch.arange(
            time_size,
            device=patch_tokens.device,
            dtype=torch.float32,
        ).view(1, time_size).expand(batch_size, -1)

        dt_steps = (
            time_index[:, 1:] - time_index[:, :-1]
        ).clamp_min(self.eps)

        total_duration = dt_steps.sum(dim=1)

        # gamma[:, 0] does not correspond to a used delta-f step.
        step_gamma = gamma[:, 1:]

        finite_sequence = torch.isfinite(step_gamma).all(dim=1)

        positive_step_gamma = torch.relu(
            torch.where(
                torch.isfinite(step_gamma),
                step_gamma,
                torch.zeros_like(step_gamma),
            )
        )

        positive_mass = positive_step_gamma.sum(dim=1)

        # New scale:
        #
        # lambda_hat = sum_k ReLU(gamma_k) / sum_k delta_t_k
        normalization_valid = (
            finite_sequence
            & (positive_mass > self.eps * total_duration)
        )

        safe_mass = torch.where(
            normalization_valid,
            positive_mass,
            torch.ones_like(positive_mass),
        )

        # Numerically stable equivalent of:
        #
        # alpha_k = gamma_k^+ / (lambda_hat * delta_t_k)
        # delta_f_k = alpha_k * delta_t_k
        #
        # Therefore:
        #
        # delta_f_k =
        #     gamma_k^+ * total_duration / sum_j gamma_j^+
        adaptive_delta_f = (
            positive_step_gamma
            * total_duration.unsqueeze(1)
            / safe_mass.unsqueeze(1)
        )

        # Retain a raw-time residual while preserving the total span:
        #
        # delta_f = (1 - beta) * delta_t + beta * adaptive_delta_f
        #
        # beta = 0 gives raw time; beta = 1 recovers the original V2.
        candidate_delta_f = (
            (1.0 - self.mix_beta) * dt_steps
            + self.mix_beta * adaptive_delta_f
        )

        # If the corrected observation variation is unreliable,
        # safely fall back to uniform raw time:
        #
        # alpha_k = 1
        # delta_f_k = delta_t_k
        delta_f_steps = torch.where(
            normalization_valid.unsqueeze(1),
            candidate_delta_f,
            dt_steps,
        )

        temporal_coordinate = torch.cat(
            (
                torch.zeros(
                    batch_size,
                    1,
                    device=patch_tokens.device,
                    dtype=torch.float32,
                ),
                torch.cumsum(delta_f_steps, dim=1),
            ),
            dim=1,
        )

        return torch.nan_to_num(
            temporal_coordinate,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )


__all__ = ["VideoSTPEV2"]
