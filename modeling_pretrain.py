import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial

from modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

from stpe_rope import (
    VideoSTPE,
    build_3d_coordinates,
    build_4d_coordinates,
    validate_pos_mode,
)
from temporal_pe_v2 import VideoSTPEV2
from video_rope import BASELINE_POS_MODES, baseline_coordinate_axes, build_baseline_coordinates


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'pretrain_videomae_small_patch16_224',
    'pretrain_videomae_base_patch16_224',
    'pretrain_videomae_large_patch16_224',
    'pretrain_videomae_huge_patch16_224',
]


class PretrainVisionTransformerEncoder(nn.Module):
    """VideoMAE pretraining encoder."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        init_values=None,
        tubelet_size=2,
        use_checkpoint=False,
        use_learnable_pos_emb=False,
        num_frames=16,
        pos_mode="original",
        rope_axis_dims=(20, 20, 24),
        rope_theta=10000.0,
        stpe_window_size=5,
        stpe_noise_mode="db4",
        stpe_mix_beta=1.0,
        rope_rotary_dim=64,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
        )

        num_patches = self.patch_embed.num_patches

        self.use_checkpoint = use_checkpoint
        self.pos_mode = validate_pos_mode(pos_mode)
        self.rope_axis_dims = tuple(rope_axis_dims)
        self.rope_theta = float(rope_theta)

        # HWF, HWF-V2 and HWFT require an observation-dependent f coordinate.
        if self.pos_mode == "hwf_v2_rope":
            self.stpe = VideoSTPEV2(
                window_size=stpe_window_size,
                noise_mode=stpe_noise_mode,
                mix_beta=stpe_mix_beta,
            )
        elif self.pos_mode in ("hwf_rope", "hwft_rope"):
            self.stpe = VideoSTPE(
                window_size=stpe_window_size,
                noise_mode=stpe_noise_mode,
            )
        else:
            self.stpe = None

        # Retain the original VideoMAE positional embedding for original mode.
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, embed_dim)
            )
        else:
            self.pos_embed = get_sinusoid_encoding_table(
                num_patches,
                embed_dim,
            )

        dpr = [
            value.item()
            for value in torch.linspace(0, drop_path_rate, depth)
        ]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[index],
                norm_layer=norm_layer,
                init_values=init_values,
                rope_axis_dims=self.rope_axis_dims,
                rope_theta=self.rope_theta,
                rope_mode=self.pos_mode,
                rope_rotary_dim=rope_rotary_dim,
            )
            for index in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        self.head = (
            nn.Linear(embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes

        self.head = (
            nn.Linear(self.embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def forward_features(self, x, mask):
        """
        Args:
            x:
                Input video [B, C, frames, height, width].

            mask:
                Boolean mask [B, number_of_tokens].
                True means masked, False means visible.

        Returns:
            x_vis:
                Encoded visible tokens [B, N_visible, encoder_dim].

            full_coords:
                Full HWT/HWF coordinate tensor [B, N_total, 3].
                None when pos_mode == "original".
        """

        # Patch embedding is computed exactly once.
        x = self.patch_embed(x)

        batch_size, num_tokens, channels = x.shape
        time_size, height, width = self.patch_embed.grid_size

        mask = mask.to(
            device=x.device,
            dtype=torch.bool,
        )

        expected_mask_shape = (batch_size, num_tokens)

        if tuple(mask.shape) != expected_mask_shape:
            raise ValueError(
                "mask must have shape {}, got {}".format(
                    expected_mask_shape,
                    tuple(mask.shape),
                )
            )

        rope_coords = None
        full_coords = None

        if self.pos_mode == "original":
            # Original VideoMAE encoder:
            # add the fixed/learnable positional embedding to all patch tokens.
            x = (
                x
                + self.pos_embed
                .type_as(x)
                .to(x.device)
                .clone()
                .detach()
            )

        elif self.pos_mode in BASELINE_POS_MODES:
            full_coords = build_baseline_coordinates(
                self.pos_mode, batch_size, (time_size, height, width), x.device
            )
            rope_coords = full_coords[~mask].reshape(
                batch_size, -1, full_coords.shape[-1]
            )

        elif self.pos_mode == "hwt_rope":
            # Raw temporal coordinate t = 0, 1, ..., T-1.
            temporal_coordinate = torch.arange(
                time_size,
                device=x.device,
                dtype=torch.float32,
            )

            temporal_coordinate = (
                temporal_coordinate
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

            # Build the complete HWT coordinates once.
            full_coords = build_3d_coordinates(
                batch_size,
                (time_size, height, width),
                temporal_coordinate,
                x.device,
            )

            # Encoder processes only visible tokens.
            rope_coords = full_coords[~mask].reshape(
                batch_size,
                -1,
                3,
            )

        elif self.pos_mode in ("hwf_rope", "hwf_v2_rope"):
            x_grid = x.reshape(
                batch_size,
                time_size,
                height,
                width,
                channels,
            )

            mask_grid = mask.reshape(
                batch_size,
                time_size,
                height,
                width,
            )

            # VideoSTPE internally selects only the visible tokens according
            # to mask_grid. Masked patch features do not participate in f.
            #
            # f is calculated exactly once here and then reused by both
            # Encoder and Decoder.
            temporal_coordinate = self.stpe(
                x_grid.detach(),
                masked_pos=mask_grid,
            )

            # Build complete HWF coordinates for all visible and masked
            # locations. The f value at a temporal step is obtained only
            # from visible tokens at that step.
            full_coords = build_3d_coordinates(
                batch_size,
                (time_size, height, width),
                temporal_coordinate,
                x.device,
            )

            # Encoder receives only visible HWF coordinates.
            rope_coords = full_coords[~mask].reshape(
                batch_size,
                -1,
                3,
            )

        elif self.pos_mode == "hwft_rope":
            x_grid = x.reshape(
                batch_size,
                time_size,
                height,
                width,
                channels,
            )

            mask_grid = mask.reshape(
                batch_size,
                time_size,
                height,
                width,
            )

            # f is computed only from tube-mask-visible tokens.
            observation_coordinate = self.stpe(
                x_grid.detach(),
                masked_pos=mask_grid,
            )

            # Complete coordinates are ordered as (h, w, f, t).
            full_coords = build_4d_coordinates(
                batch_size,
                (time_size, height, width),
                observation_coordinate,
                x.device,
            )

            # Encoder receives only visible HWFT coordinates.
            rope_coords = full_coords[~mask].reshape(
                batch_size,
                -1,
                4,
            )

        # Encoder receives only visible patch tokens.
        x_vis = x[~mask].reshape(
            batch_size,
            -1,
            channels,
        )

        if self.use_checkpoint:
            for block in self.blocks:
                if rope_coords is None:
                    x_vis = checkpoint.checkpoint(
                        block,
                        x_vis,
                    )
                else:
                    x_vis = checkpoint.checkpoint(
                        block,
                        x_vis,
                        rope_coords,
                    )
        else:
            for block in self.blocks:
                x_vis = block(
                    x_vis,
                    rope_coords,
                )

        x_vis = self.norm(x_vis)

        return x_vis, full_coords

    def forward(
        self,
        x,
        mask,
        return_full_coords=False,
    ):
        x, full_coords = self.forward_features(
            x,
            mask,
        )

        x = self.head(x)

        if return_full_coords:
            return x, full_coords

        return x


class PretrainVisionTransformerDecoder(nn.Module):
    """VideoMAE pretraining decoder."""

    def __init__(
        self,
        patch_size=16,
        num_classes=768,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        init_values=None,
        num_patches=196,
        tubelet_size=2,
        use_checkpoint=False,
        rope_axis_dims=(20, 20, 24),
        rope_theta=10000.0,
        rope_mode="hwt_rope",
        rope_rotary_dim=64,
    ):
        super().__init__()

        self.num_classes = num_classes

        assert (
            num_classes
            == 3 * tubelet_size * patch_size ** 2
        )

        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint

        self.rope_axis_dims = tuple(rope_axis_dims)
        self.rope_theta = float(rope_theta)
        self.rope_mode = rope_mode

        dpr = [
            value.item()
            for value in torch.linspace(0, drop_path_rate, depth)
        ]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[index],
                norm_layer=norm_layer,
                init_values=init_values,
                rope_axis_dims=self.rope_axis_dims,
                rope_theta=self.rope_theta,
                rope_mode=self.rope_mode,
                rope_rotary_dim=rope_rotary_dim,
            )
            for index in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        self.head = (
            nn.Linear(embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(
        self,
        num_classes,
        global_pool='',
    ):
        self.num_classes = num_classes

        self.head = (
            nn.Linear(self.embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def forward(
        self,
        x,
        return_token_num,
        rope_coords=None,
    ):
        """
        Args:
            x:
                Decoder tokens in [visible, masked] order.
                Shape [B, N_total, decoder_dim].

            return_token_num:
                Number of masked tokens. The masked tokens are stored at
                the end of x.

            rope_coords:
                Coordinates in exactly the same [visible, masked] order.
                Shape [B, N_total, 3].
        """

        if rope_coords is not None:
            expected_coord_axes = (
                baseline_coordinate_axes(self.rope_mode)
                if self.rope_mode in BASELINE_POS_MODES
                else len(self.rope_axis_dims)
            )
            expected_coords_shape = (
                x.shape[0],
                x.shape[1],
                expected_coord_axes,
            )

            if tuple(rope_coords.shape) != expected_coords_shape:
                raise ValueError(
                    "Decoder rope_coords must have shape {}, got {}".format(
                        expected_coords_shape,
                        tuple(rope_coords.shape),
                    )
                )

        if self.use_checkpoint:
            for block in self.blocks:
                if rope_coords is None:
                    x = checkpoint.checkpoint(
                        block,
                        x,
                    )
                else:
                    x = checkpoint.checkpoint(
                        block,
                        x,
                        rope_coords,
                    )
        else:
            for block in self.blocks:
                x = block(
                    x,
                    rope_coords,
                )

        if return_token_num > 0:
            # Mask tokens are stored at the end of the decoder sequence.
            x = self.head(
                self.norm(
                    x[:, -return_token_num:]
                )
            )
        else:
            x = self.head(
                self.norm(x)
            )

        return x


class PretrainVisionTransformer(nn.Module):
    """Complete VideoMAE pretraining model."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        encoder_in_chans=3,
        encoder_num_classes=0,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_num_classes=1536,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        init_values=0.,
        use_learnable_pos_emb=False,
        use_checkpoint=False,
        tubelet_size=2,
        num_frames=16,
        pos_mode="original",
        rope_axis_dims=(20, 20, 24),
        rope_theta=10000.0,
        stpe_window_size=5,
        stpe_noise_mode="db4",
        stpe_mix_beta=1.0,
        rope_rotary_dim=64,
        num_classes=0,
        in_chans=0,
    ):
        super().__init__()

        self.pos_mode = validate_pos_mode(pos_mode)

        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            num_frames=num_frames,
            pos_mode=self.pos_mode,
            rope_axis_dims=rope_axis_dims,
            rope_theta=rope_theta,
            stpe_window_size=stpe_window_size,
            stpe_noise_mode=stpe_noise_mode,
            stpe_mix_beta=stpe_mix_beta,
            rope_rotary_dim=rope_rotary_dim,
            use_checkpoint=use_checkpoint,
            use_learnable_pos_emb=use_learnable_pos_emb,
        )

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            rope_axis_dims=rope_axis_dims,
            rope_theta=rope_theta,
            rope_mode=self.pos_mode,
            rope_rotary_dim=rope_rotary_dim,
        )

        self.encoder_to_decoder = nn.Linear(
            encoder_embed_dim,
            decoder_embed_dim,
            bias=False,
        )

        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, decoder_embed_dim)
        )

        # This positional embedding is retained only for original mode.
        # It is not added in hwt_rope or hwf_rope mode.
        self.pos_embed = get_sinusoid_encoding_table(
            self.encoder.patch_embed.num_patches,
            decoder_embed_dim,
        )

        trunc_normal_(
            self.mask_token,
            std=.02,
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def get_num_layers(self):
        return (
            self.encoder.get_num_layers()
            + self.decoder.get_num_layers()
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed',
            'cls_token',
            'mask_token',
        }

    def forward(self, x, mask):
        """
        Complete pretraining forward pass.

        HWT:
            Build HWT once.
            Encoder uses visible HWT coordinates.
            Decoder uses visible and masked HWT coordinates.

        HWF:
            Compute f once from visible patch tokens.
            Build HWF once.
            Encoder uses visible HWF coordinates.
            Decoder uses visible and masked HWF coordinates.

        Original:
            Preserve original additive sinusoidal positional embeddings.
        """

        # Encoder computes patch embeddings and temporal coordinates once.
        x_vis, full_coords = self.encoder(
            x,
            mask,
            return_full_coords=True,
        )

        # [B, N_visible, encoder_dim]
        x_vis = self.encoder_to_decoder(x_vis)

        # [B, N_visible, decoder_dim]
        batch_size, _, decoder_dim = x_vis.shape

        mask = mask.to(
            device=x_vis.device,
            dtype=torch.bool,
        )

        expected_mask_shape = (
            batch_size,
            self.encoder.patch_embed.num_patches,
        )

        if tuple(mask.shape) != expected_mask_shape:
            raise ValueError(
                "mask must have shape {}, got {}".format(
                    expected_mask_shape,
                    tuple(mask.shape),
                )
            )

        masked_counts = mask.sum(dim=1)

        if not torch.all(masked_counts == masked_counts[0]):
            raise ValueError(
                "Every sample in a batch must have the same "
                "number of masked tokens"
            )

        num_mask = int(masked_counts[0].item())

        # All masked positions share one learnable base mask token.
        mask_tokens = self.mask_token.expand(
            batch_size,
            num_mask,
            decoder_dim,
        )

        if self.pos_mode == "original":
            # Original VideoMAE decoder behavior:
            # add fixed sinusoidal PE to visible features and mask tokens.
            expand_pos_embed = (
                self.pos_embed
                .expand(batch_size, -1, -1)
                .type_as(x_vis)
                .to(x_vis.device)
                .clone()
                .detach()
            )

            pos_embed_vis = expand_pos_embed[~mask].reshape(
                batch_size,
                -1,
                decoder_dim,
            )

            pos_embed_mask = expand_pos_embed[mask].reshape(
                batch_size,
                -1,
                decoder_dim,
            )

            x_full = torch.cat(
                [
                    x_vis + pos_embed_vis,
                    mask_tokens + pos_embed_mask,
                ],
                dim=1,
            )

            decoder_coords = None

        elif self.pos_mode in BASELINE_POS_MODES or self.pos_mode in (
            "hwt_rope",
            "hwf_rope",
            "hwft_rope",
            "hwf_v2_rope",
        ):
            if full_coords is None:
                raise RuntimeError(
                    "{} requires full RoPE coordinates".format(
                        self.pos_mode
                    )
                )

            # Decoder tokens are arranged as:
            # [all visible tokens, all masked tokens].
            #
            # Therefore the coordinates must be rearranged in exactly
            # the same order.
            coordinate_axes = int(full_coords.shape[-1])

            coords_vis = full_coords[~mask].reshape(
                batch_size,
                -1,
                coordinate_axes,
            )

            coords_mask = full_coords[mask].reshape(
                batch_size,
                -1,
                coordinate_axes,
            )

            decoder_coords = torch.cat(
                [
                    coords_vis,
                    coords_mask,
                ],
                dim=1,
            )

            # No sinusoidal positional embedding is added in RoPE modes.
            x_full = torch.cat(
                [
                    x_vis,
                    mask_tokens,
                ],
                dim=1,
            )

        else:
            raise RuntimeError(
                "Unexpected positional mode: {}".format(
                    self.pos_mode
                )
            )

        # HWT/HWF coordinates are supplied to every decoder attention block.
        x = self.decoder(
            x_full,
            num_mask,
            rope_coords=decoder_coords,
        )

        # [B, N_mask, 3 * tubelet_size * patch_size * patch_size]
        return x


@register_model
def pretrain_videomae_small_patch16_224(
    pretrained=False,
    **kwargs,
):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=192,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm,
            eps=1e-6,
        ),
        **kwargs,
    )

    model.default_cfg = _cfg()

    if pretrained:
        loaded_checkpoint = torch.load(
            kwargs["init_ckpt"],
            map_location="cpu",
        )

        model.load_state_dict(
            loaded_checkpoint["model"]
        )

    return model


@register_model
def pretrain_videomae_base_patch16_224(
    pretrained=False,
    **kwargs,
):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm,
            eps=1e-6,
        ),
        **kwargs,
    )

    model.default_cfg = _cfg()

    if pretrained:
        loaded_checkpoint = torch.load(
            kwargs["init_ckpt"],
            map_location="cpu",
        )

        model.load_state_dict(
            loaded_checkpoint["model"]
        )

    return model


@register_model
def pretrain_videomae_large_patch16_224(
    pretrained=False,
    **kwargs,
):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        decoder_num_classes=1536,
        decoder_embed_dim=512,
        decoder_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm,
            eps=1e-6,
        ),
        **kwargs,
    )

    model.default_cfg = _cfg()

    if pretrained:
        loaded_checkpoint = torch.load(
            kwargs["init_ckpt"],
            map_location="cpu",
        )

        model.load_state_dict(
            loaded_checkpoint["model"]
        )

    return model


@register_model
def pretrain_videomae_huge_patch16_224(
    pretrained=False,
    **kwargs,
):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        decoder_num_classes=1536,
        decoder_embed_dim=640,
        decoder_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm,
            eps=1e-6,
        ),
        **kwargs,
    )

    model.default_cfg = _cfg()

    if pretrained:
        loaded_checkpoint = torch.load(
            kwargs["init_ckpt"],
            map_location="cpu",
        )

        model.load_state_dict(
            loaded_checkpoint["model"]
        )

    return model
