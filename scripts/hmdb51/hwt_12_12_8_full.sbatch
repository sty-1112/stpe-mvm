#!/usr/bin/env bash
#SBATCH --partition=cse_24h
#SBATCH --reservation=cse_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:H200:2
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --exclude=chpc-gpu033
set -euo pipefail

: "${REPO_DIR:?}" "${PYTHON_BIN:?}" "${RUN_ROOT:?}" "${SPLIT:?}"
case "$SPLIT" in
  1|2|3) ;;
  *) echo "Invalid split: $SPLIT" >&2; exit 1 ;;
esac

cd "$REPO_DIR"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
unset CUDA_LAUNCH_BLOCKING

data_dir=/project/cse/b167368/sty/stpe-mvm/data/hmdb51/lists/split${SPLIT}
for name in train val test; do
  test -s "$data_dir/$name.csv" || {
    echo "Missing: $data_dir/$name.csv" >&2
    exit 1
  }
done

"$PYTHON_BIN" - <<'PY'
import os
import torch
n = torch.cuda.device_count()
print(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}; visible GPUs={n}", flush=True)
assert n == 2, f"Expected two allocated GPUs, got {n}"
print([torch.cuda.get_device_name(i) for i in range(n)], flush=True)
PY

run_dir="$RUN_ROOT/split${SPLIT}"
pt_dir="$run_dir/pretrain"
ft_dir="$run_dir/finetune"
ev_dir="$run_dir/evaluation"
mkdir -p "$pt_dir" "$ft_dir" "$ev_dir"

pt_ckpt="$pt_dir/checkpoint-4799.pth"
ft_last="$ft_dir/checkpoint-49.pth"
ft_best="$ft_dir/checkpoint-best.pth"

# torchrun 负责创建 rank 0/1；避免 Slurm 的单 task rank 覆盖它们。
run_ddp() {
  env -u SLURM_PROCID -u SLURM_LOCALID -u SLURM_NTASKS \
    "$PYTHON_BIN" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=2 "$@"
}

echo "START split=$SPLIT job=$SLURM_JOB_ID node=$SLURM_NODELIST run=$run_dir"

if [[ ! -s "$pt_ckpt" ]]; then
  echo "STAGE pretrain: 4800 epochs, global batch 192"
  run_ddp run_mae_pretraining.py \
    --model pretrain_videomae_small_patch16_224 \
    --pos_mode hwt_rope --rope_axis_dims 12 12 8 \
    --data_path "$data_dir/train.csv" \
    --mask_type tube --mask_ratio 0.9 \
    --num_frames 16 --sampling_rate 2 \
    --batch_size 96 --epochs 4800 --warmup_epochs 40 \
    --lr 1.5e-4 --opt adamw --opt_betas 0.9 0.95 \
    --num_workers 8 --pin_mem --save_ckpt_freq 100 --auto_resume \
    --output_dir "$pt_dir" --log_dir "$pt_dir"
fi
test -s "$pt_ckpt" || { echo "Missing: $pt_ckpt" >&2; exit 1; }

if [[ ! -s "$ft_last" || ! -s "$ft_best" ]]; then
  echo "STAGE finetune: 50 epochs, global batch 128"
  run_ddp run_class_finetuning.py \
    --model vit_small_patch16_224 --data_set HMDB51 --nb_classes 51 \
    --pos_mode hwt_rope --rope_axis_dims 12 12 8 \
    --data_path "$data_dir" --finetune "$pt_ckpt" \
    --batch_size 32 --update_freq 2 --num_sample 2 \
    --num_frames 16 --sampling_rate 2 \
    --input_size 224 --short_side_size 224 \
    --epochs 50 --warmup_epochs 5 --lr 1e-3 \
    --opt adamw --opt_betas 0.9 0.999 --weight_decay 0.05 \
    --num_workers 8 --pin_mem --save_ckpt_freq 10 --auto_resume \
    --test_num_segment 10 --test_num_crop 3 --dist_eval \
    --output_dir "$ft_dir" --log_dir "$ft_dir"
fi
test -s "$ft_last" && test -s "$ft_best" || {
  echo "Finetuning checkpoint missing: $ft_last or $ft_best" >&2
  exit 1
}

if [[ ! -s "$ev_dir/log.txt" ]] ||
   ! grep -q '"Final top-1"' "$ev_dir/log.txt"; then
  echo "STAGE evaluation: $ft_best"
  run_ddp run_class_finetuning.py \
    --model vit_small_patch16_224 --data_set HMDB51 --nb_classes 51 \
    --pos_mode hwt_rope --rope_axis_dims 12 12 8 \
    --data_path "$data_dir" --resume "$ft_best" --eval \
    --batch_size 32 --num_sample 2 \
    --num_frames 16 --sampling_rate 2 \
    --input_size 224 --short_side_size 224 \
    --num_workers 8 --pin_mem \
    --test_num_segment 10 --test_num_crop 3 --dist_eval \
    --output_dir "$ev_dir" --no_auto_resume
fi

test -s "$ev_dir/log.txt" &&
  grep -q '"Final top-1"' "$ev_dir/log.txt" || {
    echo "Evaluation metrics missing: $ev_dir/log.txt" >&2
    exit 1
  }

echo "COMPLETE split=$SPLIT metrics=$ev_dir/log.txt"
grep '"Final top-1"' "$ev_dir/log.txt" | tail -n 1
