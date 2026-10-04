#!/usr/bin/env bash
#SBATCH --job-name=hwt_diag_s1
#SBATCH --partition=cse_24h
#SBATCH --reservation=cse_gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --exclude=chpc-gpu033
#SBATCH --output=/project/cse/b167368/sty/stpe-mvm/logs/hmdb51/hwt_diag_s1_%j.out
#SBATCH --error=/project/cse/b167368/sty/stpe-mvm/logs/hmdb51/hwt_diag_s1_%j.err
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHON_BIN=/project/cse/b167368/sty/stpe-video/envs/videomae/bin/python
export OMP_NUM_THREADS=1 CUDA_LAUNCH_BLOCKING=1 PYTHONFAULTHANDLER=1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
unset MASTER_ADDR RANK WORLD_SIZE LOCAL_RANK

split_dir=/project/cse/b167368/sty/stpe-mvm/data/hmdb51/lists/split1
run_dir=/project/cse/b167368/sty/stpe-mvm/runs/hwt_diag_split1_${SLURM_JOB_ID}
mkdir -p "$run_dir"
test -s "$split_dir/train.csv" || { echo "Missing: $split_dir/train.csv" >&2; exit 1; }
echo "node=$SLURM_NODELIST job=$SLURM_JOB_ID output=$run_dir"
nvidia-smi -L

srun --ntasks=2 --ntasks-per-node=2 --gpu-bind=none --kill-on-bad-exit=1 \
  "$PYTHON_BIN" run_mae_pretraining.py \
  --model pretrain_videomae_small_patch16_224 \
  --pos_mode hwt_rope --rope_axis_dims 12 12 8 \
  --mask_type tube --mask_ratio 0.9 \
  --data_path "$split_dir/train.csv" \
  --batch_size 96 --num_frames 16 --sampling_rate 4 \
  --epochs 2 --warmup_epochs 1 --save_ckpt_freq 2 \
  --opt adamw --opt_betas 0.9 0.95 --num_workers 6 \
  --output_dir "$run_dir" --log_dir "$run_dir" --no_auto_resume

echo "DIAGNOSTIC_COMPLETE"
