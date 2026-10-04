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

## Five jobs, three splits per 72h allocation

`scripts/hmdb51/pe_baselines_3splits_72h.sbatch` takes exactly one PE mode
as its argument. Each job runs split1 pretrain -> finetune -> independent
evaluation, then split2, then split3. It calls the single-split helper with
`bash` inside the same allocation; the helper's 24h SBATCH header is a
comment in this context. It does not submit nested jobs.

The resource request is 72h, one node, one Slurm task, 32 CPUs, 128G host
memory and two H200s. The script retains reservation `cse_gpu` and the
previous temporary exclusion of `chpc-gpu033`. Override reservation or
exclude at submission if the site's 72h queue requires it. Project notes
verify `cse_24h` only, so the new script deliberately leaves the partition
to `sbatch --partition`. Verify the actual partition and whether it admits
the requested reservation, GPU type and time limit on the cluster.

Default paths are the successful project paths:

| Variable | Default / requirement |
| --- | --- |
| `REPO_DIR` | `/project/cse/b167368/sty/stpe-mvm/VideoMAE` |
| `PYTHON_BIN` | `/project/cse/b167368/sty/stpe-video/envs/videomae/bin/python` |
| `DATA_ROOT` | `/project/cse/b167368/sty/stpe-mvm/data/hmdb51/lists` |
| `RUN_ROOT` | Required; export one new root shared by the five jobs |

Keep Slurm's `CUDA_VISIBLE_DEVICES`. Two ranks are created only by
`torchrun --standalone --nnodes=1 --nproc_per_node=2`, after removing
`SLURM_PROCID`, `SLURM_LOCALID` and `SLURM_NTASKS` from its environment.
Per-rank DataLoader workers remain 8 and OMP threads remain 1. The
4800/50-epoch protocol, per-rank batches 96/32, FT update_freq=2,
16 frames with sampling_rate=2 and 10-segment/3-crop evaluation are
unchanged. The PE defaults are `(h,w,t/f)=(24,24,16)`, rotary dimension
64, theta 10000, TAD gamma 1, VideoRoPE spacing 2, and VideoRoPE(f)
estimator v2/window 5/db4/beta 1. Set the corresponding environment
variables described in the single-split helper before submitting if needed.

On the cluster, preserve local changes before switching to `baseline`
and pulling it. Use a new RUN_ROOT for this experiment. First inspect
the available partition names and time limits; replace
`YOUR_72H_PARTITION` below with the real name.

```bash
cd /project/cse/b167368/sty/stpe-mvm/VideoMAE
git fetch origin
git switch baseline
git pull --ff-only origin baseline

export REPO_DIR="$PWD"
export PYTHON_BIN=/project/cse/b167368/sty/stpe-video/envs/videomae/bin/python
export DATA_ROOT=/project/cse/b167368/sty/stpe-mvm/data/hmdb51/lists
export RUN_ROOT=/project/cse/b167368/sty/stpe-mvm/outputs/pe_baselines_3splits_$(date +%Y%m%d_%H%M%S)
LOG_ROOT=/project/cse/b167368/sty/stpe-mvm/logs/hmdb51
mkdir -p "$LOG_ROOT"
printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
sinfo -h -o '%P %l %G'
export PARTITION_72H=YOUR_72H_PARTITION

job_script=scripts/hmdb51/pe_baselines_3splits_72h.sbatch
sbatch_args=(--partition="$PARTITION_72H" --export=ALL
             --output="$LOG_ROOT/%x_%j.out" --error="$LOG_ROOT/%x_%j.err")
sbatch --test-only "${sbatch_args[@]}" "$job_script" vanilla_rope

sbatch "${sbatch_args[@]}" --job-name=hmdb51_vanilla_3s "$job_script" vanilla_rope
sbatch "${sbatch_args[@]}" --job-name=hmdb51_tad_3s "$job_script" tad_rope
sbatch "${sbatch_args[@]}" --job-name=hmdb51_mrope_3s "$job_script" m_rope
sbatch "${sbatch_args[@]}" --job-name=hmdb51_videorope_3s "$job_script" video_rope
sbatch "${sbatch_args[@]}" --job-name=hmdb51_videorope_f_3s "$job_script" video_rope_f
```

`sbatch --test-only` checks scheduling feasibility without submitting a
job; it does not run data/GPU preflight. Only run the five actual
submission commands after that check succeeds. Five jobs request ten
H200s in total if scheduled simultaneously; available quota may queue
them. The known HWT cost is about 15-17h per split (45-51h for three),
but the new PE methods' throughput has not been measured on H200.
The 72h request is a budget, not a guarantee of completion.

Outputs are `$RUN_ROOT/<mode>/split{1,2,3}/{pretrain,finetune,evaluation}`.
The wrapper preflights all nine lists and listed video paths (existence
and readability only; no decoding). It saves PE/protocol settings,
list/code SHA256 fingerprints, commit/local diff, script copies and Slurm
job details under `<mode>/jobs/<job_id>/`. A per-mode lock prevents two
jobs from writing to the same experiment simultaneously. Keep the source
tree unchanged while the jobs run: the fingerprint is a provenance and
resume guard, not a separate checkout of the code.

To resume after interruption or a time limit, export the **same**
RUN_ROOT and resubmit only the affected mode with the same options.
Completed stages are skipped; unfinished training uses auto_resume.
Changing a tracked training source, a list, or PE options is rejected;
use a fresh RUN_ROOT for a new trial. Each completed split must have
PT checkpoint-4799, FT checkpoint-49 and checkpoint-best with matching
PE/frame/sampling settings, plus valid merged independent evaluation
metrics. Best checkpoints use `epoch="best"` as saved by this repository.
The script stops on a failed stage or validation, before the next split.

After all three splits, `<mode>/results.json` and `results.csv` contain
split Top-1/Top-5 and their arithmetic means. Only
`splitX/evaluation/log.txt` is used, never the final-model test results
inside the finetuning log. A successful wrapper prints
`COMPLETE_3SPLITS`; also verify the Slurm state and exit code.

The orchestration smoke tests use simulated GPU checks and training
processes, real PyTorch checkpoint serialization and the actual CLI
parsers. They cover all five modes, nine-stage ordering, output isolation,
stage skipping, recovery after failure, configuration mismatch, missing
data, task-count mistakes, concurrent writes and final metric checks:

```bash
python -m pytest -q tests/test_hmdb51_72h_script.py
bash -n scripts/hmdb51/pe_baseline_full.sbatch
bash -n scripts/hmdb51/pe_baselines_3splits_72h.sbatch
```
