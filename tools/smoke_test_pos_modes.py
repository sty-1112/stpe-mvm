"""Run small original/HWT/HWF forward-backward compatibility checks."""

from functools import partial
from pathlib import Path
import sys

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modeling_finetune import VisionTransformer
from modeling_pretrain import PretrainVisionTransformer


MODES = ("original", "hwt_rope", "hwf_rope", "hwf_v2_rope")


def make_tube_mask(batch_size, time_size, height, width):
    spatial = torch.tensor(
        [[False, True], [False, True]], dtype=torch.bool
    ).reshape(1, 1, height, width)
    return spatial.expand(batch_size, time_size, height, width).reshape(batch_size, -1)


def build_pretrain_model(pos_mode):
    torch.manual_seed(7)
    return PretrainVisionTransformer(
        img_size=32,
        patch_size=16,
        encoder_embed_dim=64,
        encoder_depth=1,
        encoder_num_heads=1,
        decoder_num_classes=1536,
        decoder_embed_dim=64,
        decoder_depth=1,
        decoder_num_heads=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tubelet_size=2,
        num_frames=4,
        pos_mode=pos_mode,
        rope_axis_dims=(20, 20, 24),
        stpe_window_size=3,
    )


def build_finetune_model(pos_mode):
    torch.manual_seed(11)
    return VisionTransformer(
        img_size=32,
        patch_size=16,
        num_classes=5,
        embed_dim=64,
        depth=1,
        num_heads=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        all_frames=4,
        tubelet_size=2,
        pos_mode=pos_mode,
        rope_axis_dims=(20, 20, 24),
        stpe_window_size=3,
    )


def main():
    video = torch.randn(2, 3, 4, 32, 32)
    mask = make_tube_mask(2, 2, 2, 2)

    pretrain_models = {mode: build_pretrain_model(mode) for mode in MODES}
    pretrain_keys = [set(model.state_dict().keys()) for model in pretrain_models.values()]
    assert pretrain_keys[0] == pretrain_keys[1] == pretrain_keys[2]

    for mode, model in pretrain_models.items():
        output = model(video, mask)
        assert output.shape == (2, 4, 1536), (mode, output.shape)
        loss = output.square().mean()
        loss.backward()
        print("pretrain {}: PASS shape={} loss={:.6f}".format(
            mode, tuple(output.shape), loss.item()
        ))

    finetune_models = {mode: build_finetune_model(mode) for mode in MODES}
    finetune_keys = [set(model.state_dict().keys()) for model in finetune_models.values()]
    assert finetune_keys[0] == finetune_keys[1] == finetune_keys[2]

    for mode, model in finetune_models.items():
        output = model(video)
        assert output.shape == (2, 5), (mode, output.shape)
        loss = output.square().mean()
        loss.backward()
        print("finetune {}: PASS shape={} loss={:.6f}".format(
            mode, tuple(output.shape), loss.item()
        ))


if __name__ == "__main__":
    main()
