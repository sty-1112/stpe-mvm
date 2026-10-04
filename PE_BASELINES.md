# VideoMAE PE baselines

These are controlled adaptations of positional encodings to VideoMAE, not
reproductions of video-language-model benchmark scores. Existing `original`,
`hwt_rope`, `hwf_rope`, `hwft_rope` and `hwf_v2_rope` behavior is retained.

## Implemented modes

| CLI mode | Positions | Frequency allocation |
| --- | --- | --- |
| `vanilla_rope` | Full-grid flattened index `n = t*H*W + h*W + w` | All global RoPE frequencies use `n` |
| `tad_rope` | `n + gamma*t` | Global and temporal rotations compose on the same channels |
| `m_rope` | Grid coordinates `(h,w,t)` | Global frequencies assigned to `t`, then `h`, then `w` |
| `video_rope` | `(delta*t+h-c_h, delta*t+w-c_w, delta*t)` | Interleaved spatial high frequencies, temporal low frequencies |
| `video_rope_f` | `(delta*f+h-c_h, delta*f+w-c_w, delta*f)` | Identical frequency allocation to `video_rope` |

New baselines default to `--rope_rotary_dim 64 --rope_theta 10000`.
Any channels beyond `rope_rotary_dim` remain unchanged. The existing
`--rope_axis_dims` option keeps the `(h,w,t/f)` convention; it is irrelevant
for one-dimensional Vanilla RoPE. Legacy modes still use independent
per-axis frequencies and ignore `rope_rotary_dim`.

TAD-RoPE follows TC-LLaVA (AAAI 2025): the two rotations are equivalent to
adding position angles, not assigning them to disjoint channel groups.
`--tad_gamma` defaults to 1, matching VideoRoPE's official comparison code;
gamma=0 recovers Vanilla RoPE exactly. This adaptation retains VideoMAE's
bidirectional attention and does not add TC-LLaVA's causal attention mask.
Reference: https://arxiv.org/abs/2409.03206

M-RoPE follows Qwen2-VL (2024 technical report). Use
`--rope_axis_dims 24 24 16 --rope_rotary_dim 64` to scale the reference
128-channel allocation (48 spatial channels per axis, 32 temporal channels)
to ViT-S's 64-channel attention heads. The sum of axis dimensions must equal
the rotated dimension. Frequencies are generated once over that entire
dimension, rather than restarting at each axis as in legacy HWT/HWF.
M-RoPE assigns the first 8 (highest-frequency) rotation pairs to time,
the next 12 to height, and the last 12 to width. Adjacent channel pairing is
equivalent to Qwen's half-split pairing under a fixed channel permutation;
this adaptation uses the repository's existing adjacent-pair convention.
Reference: https://arxiv.org/abs/2409.12191

VideoRoPE follows the ICML 2025 paper and its official implementation:
Low-frequency Temporal Allocation, Diagonal Layout and Adjustable Temporal
Spacing. With axis dims `(24,24,16)`, the first 24 rotation pairs alternate
height/width, and the last 8 pairs encode time. Spatial dimensions must be
equal for this interleaving. `--temporal_spacing` defaults to 2, as in the
official code. `c_h=(H-1)//2`, `c_w=(W-1)//2` also follow that code.
The spatial coordinates include the temporal offset, so corresponding
patches in successive frames move along all three positional axes.
There are no text tokens or causal-mask changes in this VideoMAE adaptation.
References: https://arxiv.org/abs/2502.05173 and
https://github.com/Wiselnn570/VideoRoPE

VideoRoPE(f) replaces time in all three diagonal coordinates. It calculates
f once from detached unpositioned PatchEmbed features; during pretraining,
only tube-mask-visible features contribute, and the encoder and decoder
reuse the same coordinate. Finetuning and evaluation use all observed
tokens. Non-tube pretraining masks are rejected by the existing estimator.
Continuous f is retained in float32 even under mixed precision.

`--stpe_estimator v2` is the default for this new mode and reuses the
existing `VideoSTPEV2` implementation: f's total span equals raw time's
span. `--stpe_mix_beta 1` uses the adaptive coordinate; beta=0 exactly
recovers VideoRoPE(t), and intermediate values mix adaptive and raw time.
`--stpe_estimator v1` reuses the older HWF `VideoSTPE` estimator, which
does not guarantee the same total span and ignores the mixing parameter.
Existing HWF modes retain their original estimator selection and defaults.
Record the estimator, window size, noise mode, beta and temporal spacing
when reporting this row. Change only the temporal coordinate when comparing
VideoRoPE(t) and VideoRoPE(f); keep the same rotary dimension, axis budget,
theta, spacing, training protocol and evaluation procedure.

Positions are created on the complete tubelet grid before selecting visible
tokens. The decoder receives coordinates reordered to `[visible, masked]`;
neither encoder nor decoder renumbers the positions after masking.
For 16 sampled frames and tubelet size 2, temporal positions are `0..7`.

## CPU smoke tests

With the project's PyTorch and timm dependencies plus pytest installed:

```bash
python -m pytest -q tests/test_pe_baselines.py tests/test_stpe_rope.py
```

The smoke models use a 32x32 image grid and shallow networks, retaining
ViT-S's 64-channel encoder and decoder attention heads. Tests exercise real
pretraining reconstruction, classification, backward passes, activation
checkpointing, decoder coordinate ordering, pretrain-to-finetune weight
transfer, checkpoint save/load and both actual CLI parsers. They do not
validate CUDA, NCCL, dataset decoding, full-run resource use or accuracy.
The tests also construct the registered production ViT-S factories with
training-entrypoint options, exercise CPU bfloat16 autocast, check that
masked input changes do not affect f, and verify beta=0 recovers VideoRoPE(t)
in both reconstruction and classification. Both v1 and v2 estimators are
covered; the v2 coordinate is checked for span preservation.

## HMDB51 full pipeline

`scripts/hmdb51/pe_baseline_full.sbatch` preserves the successful HWT
protocol: 32 CPUs, 128G host memory, two H200s, one Slurm task with torchrun,
4800 pretraining epochs, 50 finetuning epochs, sampling rate 2, tube mask
ratio 0.9, and independent best-checkpoint evaluation. Pretraining receives
`splitX/train.csv`; classification receives the `splitX` directory.

Set `REPO_DIR`, `PYTHON_BIN`, `RUN_ROOT`, `SPLIT` and `POS_MODE` explicitly
before submitting. Each mode has its own output subdirectory. Different
hyperparameter trials must use different RUN_ROOTs; a saved PE argument
manifest prevents silently resuming a run with different PE settings.

```bash
export REPO_DIR=/project/cse/b167368/sty/stpe-mvm/VideoMAE
export PYTHON_BIN=/project/cse/b167368/sty/stpe-video/envs/videomae/bin/python
export RUN_ROOT=/project/cse/b167368/sty/stpe-mvm/outputs/pe_baselines_trial1
export SPLIT=1 POS_MODE=vanilla_rope
mkdir -p /project/cse/b167368/sty/stpe-mvm/logs/hmdb51
sbatch --export=ALL \
  --output=/project/cse/b167368/sty/stpe-mvm/logs/hmdb51/%x_%j.out \
  --error=/project/cse/b167368/sty/stpe-mvm/logs/hmdb51/%x_%j.err \
  scripts/hmdb51/pe_baseline_full.sbatch
```

No cluster jobs are submitted by implementation or smoke testing.
