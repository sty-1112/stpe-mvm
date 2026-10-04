#!/usr/bin/env bash
# One allocation per PE: split1 (PT->FT->eval), then split2, then split3.
# Choose the actual 72h partition with sbatch --partition=...; it is site-specific.
#SBATCH --reservation=cse_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:H200:2
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=chpc-gpu033
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: sbatch --partition=<72h-partition> $0 <PE-mode>" >&2
  exit 2
fi
export POS_MODE="$1"
case "$POS_MODE" in
  vanilla_rope|tad_rope|m_rope|video_rope|video_rope_f) ;;
  *) echo "Unsupported PE mode: $POS_MODE" >&2; exit 2 ;;
esac

export REPO_DIR=${REPO_DIR:-/project/cse/b167368/sty/stpe-mvm/VideoMAE}
export PYTHON_BIN=${PYTHON_BIN:-/project/cse/b167368/sty/stpe-video/envs/videomae/bin/python}
export DATA_ROOT=${DATA_ROOT:-/project/cse/b167368/sty/stpe-mvm/data/hmdb51/lists}
: "${RUN_ROOT:?Export one RUN_ROOT shared by the five jobs; reuse it only to resume this experiment}"
export RUN_ROOT
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
unset CUDA_LAUNCH_BLOCKING

test -d "$REPO_DIR" || { echo "Missing repo: $REPO_DIR" >&2; exit 1; }
test -x "$PYTHON_BIN" || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
cd "$REPO_DIR"
single_split_script="$REPO_DIR/scripts/hmdb51/pe_baseline_full.sbatch"
summary_script="$REPO_DIR/scripts/hmdb51/summarize_pe_baselines.py"
for path in "$single_split_script" "$summary_script"; do
  test -s "$path" || { echo "Missing helper: $path" >&2; exit 1; }
done
bash -n "$single_split_script"

# Fail before starting training if the allocation creates too many Slurm tasks.
if [[ "${SLURM_NTASKS:-1}" != 1 || "${SLURM_JOB_NUM_NODES:-1}" != 1 ]]; then
  echo "Expected one node and one Slurm task; torchrun creates the two ranks" >&2
  exit 1
fi

mode_dir="$RUN_ROOT/$POS_MODE"
mkdir -p "$mode_dir/jobs"
command -v flock >/dev/null || { echo "flock is required to prevent concurrent resume" >&2; exit 1; }
exec 9>"$mode_dir/pipeline.lock"
flock -n 9 || { echo "Another job is using $mode_dir" >&2; exit 1; }

job_tag=${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)_$$}
job_dir="$mode_dir/jobs/$job_tag"
mkdir -p "$job_dir"
current_split=preflight
trap 'rc=$?; printf "FAILED mode=%s split=%s job=%s exit=%s\n" "$POS_MODE" "$current_split" "$job_tag" "$rc" >&2; exit "$rc"' ERR

git rev-parse HEAD > "$job_dir/code_commit.txt"
git status --short > "$job_dir/code_status.txt"
git diff HEAD > "$job_dir/code_diff.patch"
cp "$single_split_script" "$job_dir/single_split_script.sbatch"
cp "$REPO_DIR/scripts/hmdb51/pe_baselines_3splits_72h.sbatch" "$job_dir/three_split_script.sbatch"
if command -v scontrol >/dev/null && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job "$SLURM_JOB_ID" > "$job_dir/slurm_job.txt"
fi

# Guard the experiment across restarts: same code content, lists and PE options.
# Read every listed video path on the allocated node, matching kinetics.py's
# whitespace-separated list format. Never rewrite or regenerate split lists.
"$PYTHON_BIN" - "$mode_dir/run_config.json" "$job_dir" <<'PYCONFIG'
import hashlib, json, os, pathlib, subprocess, sys
repo = pathlib.Path(os.environ["REPO_DIR"]).resolve()
data = pathlib.Path(os.environ["DATA_ROOT"]).resolve()
list_hashes = {}
for split in (1, 2, 3):
    for name in ("train", "val", "test"):
        path = data / f"split{split}" / f"{name}.csv"
        if not path.is_file() or not path.stat().st_size:
            raise SystemExit(f"Missing or empty list: {path}")
        lines = path.read_text().splitlines()
        if not lines or any(not line.strip() for line in lines):
            raise SystemExit(f"Empty row in list: {path}")
        for row, line in enumerate(lines, 1):
            fields = line.split()
            if len(fields) < 2:
                raise SystemExit(f"Malformed row: {path}:{row}")
            video = pathlib.Path(fields[0])
            if not video.is_file() or not os.access(video, os.R_OK):
                raise SystemExit(f"Unreadable video: {path}:{row}: {video}")
        list_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"LIST split={split} {name}: {len(lines)} rows", flush=True)
sources = [
    "video_rope.py", "stpe_rope.py", "temporal_pe_v2.py",
    "modeling_pretrain.py", "modeling_finetune.py", "run_mae_pretraining.py",
    "run_class_finetuning.py", "datasets.py", "kinetics.py", "utils.py",
    "engine_for_pretraining.py", "engine_for_finetuning.py", "optim_factory.py",
    "scripts/hmdb51/pe_baseline_full.sbatch",
]
code_hashes = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in sources}
mode = os.environ["POS_MODE"]
pe = {"pos_mode": mode, "axis_dims": os.getenv("ROPE_AXIS_DIMS", "24 24 16").split(),
      "rotary_dim": os.getenv("ROPE_ROTARY_DIM", "64"), "theta": os.getenv("ROPE_THETA", "10000")}
if mode == "tad_rope":
    pe["gamma"] = os.getenv("TAD_GAMMA", "1")
if mode in ("video_rope", "video_rope_f"):
    pe["spacing"] = os.getenv("TEMPORAL_SPACING", "2")
if mode == "video_rope_f":
    pe.update(estimator=os.getenv("STPE_ESTIMATOR", "v2"), window=os.getenv("STPE_WINDOW_SIZE", "5"),
              noise=os.getenv("STPE_NOISE_MODE", "db4"), beta=os.getenv("STPE_MIX_BETA", "1"))
config = {"schema": 1, "repo": str(repo), "python": os.environ["PYTHON_BIN"],
          "splits": [1, 2, 3], "pe": pe, "code_sha256": code_hashes, "lists_sha256": list_hashes,
          "protocol": {"pretrain_epochs": 4800, "finetune_epochs": 50, "num_frames": 16,
                       "sampling_rate": 2, "world_size": 2, "pretrain_batch_per_rank": 96,
                       "finetune_batch_per_rank": 32, "update_freq": 2, "num_sample": 2,
                       "test_segments": 10, "test_crops": 3}}
path = pathlib.Path(sys.argv[1])
if path.exists() and json.loads(path.read_text()) != config:
    raise SystemExit("Run configuration/code/data changed; use a new RUN_ROOT")
if not path.exists():
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
pathlib.Path(sys.argv[2], "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PYCONFIG

echo "START_3SPLITS mode=$POS_MODE job=$job_tag partition=${SLURM_JOB_PARTITION:-unknown} node=${SLURM_NODELIST:-unknown} run=$mode_dir"
for split in 1 2 3; do
  current_split=$split
  start_seconds=$SECONDS
  echo "BEGIN_SPLIT mode=$POS_MODE split=$split"
  # Run the helper as bash in the current allocation. Its 24h SBATCH header
  # is a comment here; no nested sbatch/srun and no additional jobs are created.
  SPLIT="$split" bash "$single_split_script"
  "$PYTHON_BIN" "$summary_script" --run-root "$RUN_ROOT" --mode "$POS_MODE" \
    --splits "$split" --validate-checkpoints --check-only
  printf 'END_SPLIT mode=%s split=%s elapsed_seconds=%s\n' "$POS_MODE" "$split" "$((SECONDS-start_seconds))"
done

current_split=summary
"$PYTHON_BIN" "$summary_script" --run-root "$RUN_ROOT" --mode "$POS_MODE"
echo "COMPLETE_3SPLITS mode=$POS_MODE job=$job_tag summary=$mode_dir/results.json"
