# VideoMAE PE baselines

These are controlled adaptations of positional encodings to VideoMAE, not
reproductions of video-language-model benchmark scores. Existing `original`,
`hwt_rope`, `hwf_rope`, `hwft_rope` and `hwf_v2_rope` behavior is retained.

## Implemented modes

| CLI mode | Positions | Frequency allocation |
| --- | --- | --- |
| `vanilla_rope` | Full-grid flattened index `n = t*H*W + h*W + w` | All global RoPE frequencies use `n` |

New baselines default to `--rope_rotary_dim 64 --rope_theta 10000`.
Any channels beyond `rope_rotary_dim` remain unchanged. The existing
`--rope_axis_dims` option keeps the `(h,w,t/f)` convention; it is irrelevant
for one-dimensional Vanilla RoPE. Legacy modes still use independent
per-axis frequencies and ignore `rope_rotary_dim`.

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
