"""Exercise the real shell pipeline with simulated GPU/training processes.

This checks orchestration and real checkpoint metadata, not CUDA or decoding.
"""

import argparse
import ast
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/hmdb51/pe_baselines_3splits_72h.sbatch"
spec = importlib.util.spec_from_file_location(
    "summarize_pe_baselines", ROOT / "scripts/hmdb51/summarize_pe_baselines.py"
)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)
MODES = summary.MODES


FAKE_PYTHON = r'''
import json, os, pathlib, shutil, sys
args = sys.argv[1:]
if args[0] == "-":
    source = sys.stdin.read()
    if "torch.cuda.device_count()" in source:
        print("SIMULATION: two allocated GPUs; no actual CUDA check")
    else:
        sys.argv = args
        exec(compile(source, "<stdin>", "exec"))
elif args[:2] == ["-m", "torch.distributed.run"]:
    assert args[2:5] == ["--standalone", "--nnodes=1", "--nproc_per_node=2"]
    assert all(key not in os.environ for key in ("SLURM_PROCID", "SLURM_LOCALID", "SLURM_NTASKS"))
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5,6"
    stage = "pretrain" if args[5] == "run_mae_pretraining.py" else ("evaluation" if "--eval" in args else "finetune")
    out = pathlib.Path(args[args.index("--output_dir") + 1])
    out.mkdir(parents=True, exist_ok=True)
    split = int(out.parent.name[-1])
    with open(os.environ["FAKE_EVENTS"], "a") as handle:
        handle.write(json.dumps({"stage": stage, "split": split, "argv": args[5:]}) + "\n")
    if stage == os.environ.get("FAIL_STAGE") and split == 1:
        raise SystemExit(7)
    templates = pathlib.Path(os.environ["FAKE_TEMPLATES"])
    if stage == "pretrain":
        shutil.copyfile(templates / "pt.pth", out / "checkpoint-4799.pth")
    elif stage == "finetune":
        shutil.copyfile(templates / "ft.pth", out / "checkpoint-49.pth")
        shutil.copyfile(templates / "best.pth", out / "checkpoint-best.pth")
        (out / "log.txt").write_text(json.dumps({"Final top-1": 99, "Final Top-5": 100}) + "\n")
    else:
        (out / "log.txt").write_text(json.dumps({"Final top-1": 40 + split, "Final Top-5": 70 + split}) + "\n")
else:
    os.execv(sys.executable, [sys.executable, *args])
'''


def prepare(tmp_path, mode="vanilla_rope"):
    video = tmp_path / "video.avi"
    video.write_bytes(b"simulation, not a decodable video")
    data = tmp_path / "lists"
    for split in (1, 2, 3):
        folder = data / f"split{split}"
        folder.mkdir(parents=True)
        for name in ("train", "val", "test"):
            (folder / f"{name}.csv").write_text(f"{video} 0\n")
    templates = tmp_path / "templates"
    templates.mkdir()
    checkpoint_args = argparse.Namespace(
        pos_mode=mode, rope_axis_dims=[24, 24, 16], rope_rotary_dim=64,
        rope_theta=10000., tad_gamma=1., temporal_spacing=2.,
        stpe_estimator="v2", stpe_window_size=5, stpe_noise_mode="db4",
        stpe_mix_beta=1., num_frames=16, sampling_rate=2,
    )
    for name, epoch in (("pt", 4799), ("ft", 49), ("best", "best")):
        torch.save({"model": {"weight": torch.ones(1)}, "epoch": epoch, "args": checkpoint_args}, templates / f"{name}.pth")
    python = tmp_path / "fake_python"
    python.write_text(f"#!{sys.executable}\n" + FAKE_PYTHON)
    python.chmod(0o755)
    env = os.environ.copy()
    for key in ("ROPE_AXIS_DIMS", "ROPE_ROTARY_DIM", "ROPE_THETA", "TAD_GAMMA", "TEMPORAL_SPACING",
                "STPE_ESTIMATOR", "STPE_WINDOW_SIZE", "STPE_NOISE_MODE", "STPE_MIX_BETA", "FAIL_STAGE"):
        env.pop(key, None)
    env.update(REPO_DIR=str(ROOT), PYTHON_BIN=str(python), DATA_ROOT=str(data),
               RUN_ROOT=str(tmp_path / "outputs"), FAKE_EVENTS=str(tmp_path / "events.jsonl"),
               FAKE_TEMPLATES=str(templates), CUDA_VISIBLE_DEVICES="5,6", SLURM_NTASKS="1",
               SLURM_JOB_NUM_NODES="1", SLURM_PROCID="0", SLURM_LOCALID="0",
               SLURM_JOB_ID="simulated", SLURM_NODELIST="simulated")
    return env


def run(env, mode="vanilla_rope"):
    return subprocess.run(["bash", str(SCRIPT), mode], env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)


def events(env):
    path = Path(env["FAKE_EVENTS"])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def parse_training_args(argv, monkeypatch):
    # Compile each actual CLI parser, without importing dataset dependencies.
    tree = ast.parse((ROOT / argv[0]).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_args")
    namespace = {"argparse": argparse, "BASELINE_POS_MODES": MODES}
    exec(compile(ast.Module(body=[function], type_ignores=[]), argv[0], "exec"), namespace)
    monkeypatch.setattr(sys, "argv", argv)
    parsed = namespace["get_args"]()
    return parsed[0] if isinstance(parsed, tuple) else parsed


@pytest.mark.parametrize("mode", MODES)
def test_five_modes_complete_and_resume(tmp_path, mode, monkeypatch):
    env = prepare(tmp_path, mode)
    result = run(env, mode)
    assert result.returncode == 0, result.stdout
    calls = events(env)
    assert [(call["split"], call["stage"]) for call in calls] == [
        (split, stage) for split in (1, 2, 3) for stage in ("pretrain", "finetune", "evaluation")
    ]
    for call in calls:
        args = parse_training_args(call["argv"], monkeypatch)
        split = call["split"]
        split_dir = Path(env["RUN_ROOT"]) / mode / f"split{split}"
        data_dir = Path(env["DATA_ROOT"]) / f"split{split}"
        assert args.pos_mode == mode and args.rope_axis_dims == [24, 24, 16]
        assert args.rope_rotary_dim == 64 and args.rope_theta == 10000
        assert args.num_frames == 16 and args.sampling_rate == 2
        assert args.num_workers == 8 and args.pin_mem
        assert args.data_path == str(data_dir / "train.csv" if call["stage"] == "pretrain" else data_dir)
        if call["stage"] == "pretrain":
            assert args.epochs == 4800 and args.batch_size == 96 and args.auto_resume
            assert args.mask_ratio == .9 and args.opt_betas == [.9, .95]
        elif call["stage"] == "finetune":
            assert args.epochs == 50 and args.batch_size == 32 and args.update_freq == 2
            assert args.finetune == str(split_dir / "pretrain/checkpoint-4799.pth")
            assert args.auto_resume and args.num_sample == 2
        else:
            assert args.eval and not args.auto_resume
            assert args.resume == str(split_dir / "finetune/checkpoint-best.pth")
        if call["stage"] != "pretrain":
            assert args.test_num_segment == 10 and args.test_num_crop == 3 and args.dist_eval
    mode_dir = Path(env["RUN_ROOT"]) / mode
    result_json = json.loads((mode_dir / "results.json").read_text())
    assert result_json["mean"] == {"top1": 42., "top5": 72.}
    assert (mode_dir / "results.csv").read_text().splitlines()[-1] == f"{mode},mean,42.0,72.0"
    manifest = json.loads((mode_dir / "run_config.json").read_text())
    assert len(manifest["lists_sha256"]) == 9 and manifest["protocol"]["world_size"] == 2
    env["SLURM_JOB_ID"] = "resumed"
    resumed = run(env, mode)
    assert resumed.returncode == 0 and len(events(env)) == 9, resumed.stdout
    env["ROPE_THETA"] = "50000"
    changed = run(env, mode)
    assert changed.returncode != 0 and "configuration/code/data changed" in changed.stdout
    assert len(events(env)) == 9


def test_failed_stage_stops_then_resumes(tmp_path):
    env = prepare(tmp_path)
    env["FAIL_STAGE"] = "finetune"
    failed = run(env)
    assert failed.returncode == 7 and "FAILED mode=vanilla_rope split=1" in failed.stdout
    assert [(call["split"], call["stage"]) for call in events(env)] == [(1, "pretrain"), (1, "finetune")]
    env.pop("FAIL_STAGE")
    resumed = run(env)
    assert resumed.returncode == 0, resumed.stdout
    assert len(events(env)) == 10 and events(env)[2]["stage"] == "finetune"


@pytest.mark.parametrize("failure", ("missing_list", "missing_video", "slurm_tasks", "mode", "concurrent"))
def test_preflight_rejects_before_training(tmp_path, failure):
    env = prepare(tmp_path)
    mode = "vanilla_rope"
    lock = None
    if failure == "missing_list":
        (Path(env["DATA_ROOT"]) / "split3/test.csv").unlink()
    elif failure == "missing_video":
        (tmp_path / "video.avi").unlink()
    elif failure == "slurm_tasks":
        env["SLURM_NTASKS"] = "2"
    elif failure == "mode":
        mode = "misspelled_rope"
    else:
        mode_dir = Path(env["RUN_ROOT"]) / mode
        mode_dir.mkdir(parents=True)
        lock = (mode_dir / "pipeline.lock").open("w")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        result = run(env, mode)
    finally:
        if lock is not None:
            lock.close()
    assert result.returncode != 0 and not events(env), result.stdout


@pytest.mark.parametrize("record", ({"Final top-1": 90, "Final Top-5": 80},
                                    {"Final top-1": float("nan"), "Final Top-5": 80},
                                    {"test_acc1": 80}, {"Final top-1": 50}))
def test_summary_rejects_invalid_metrics(tmp_path, record):
    path = tmp_path / "log.txt"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError):
        summary.read_metrics(path)


def test_summary_reads_merged_record(tmp_path):
    path = tmp_path / "log.txt"
    path.write_text('42\n[]\n{"val_acc1":99}\n{"Final top-1":41,"Final Top-5":71}\n{"incomplete":')
    assert summary.read_metrics(path) == {"top1": 41, "top5": 71}


def test_checkpoint_guard_uses_repository_best_format(tmp_path):
    env = prepare(tmp_path)
    split_dir = tmp_path / "split1"
    for stage in ("pretrain", "finetune"):
        (split_dir / stage).mkdir(parents=True)
    import shutil
    for source, target in (("pt", "pretrain/checkpoint-4799.pth"),
                           ("ft", "finetune/checkpoint-49.pth"), ("best", "finetune/checkpoint-best.pth")):
        shutil.copyfile(Path(env["FAKE_TEMPLATES"]) / f"{source}.pth", split_dir / target)
    (split_dir / "pe_args.json").write_text(json.dumps(["--pos_mode", "vanilla_rope", "--rope_rotary_dim", "64"]))
    summary.validate_checkpoints(split_dir, "vanilla_rope")
    checkpoint = torch.load(split_dir / "pretrain/checkpoint-4799.pth", weights_only=False)
    checkpoint["args"].sampling_rate = 4
    torch.save(checkpoint, split_dir / "pretrain/checkpoint-4799.pth")
    with pytest.raises(ValueError, match="sampling_rate"):
        summary.validate_checkpoints(split_dir, "vanilla_rope")


def test_allocation_header():
    source = SCRIPT.read_text()
    for directive in ("--nodes=1", "--ntasks=1", "--cpus-per-task=32", "--mem=128G",
                      "--gres=gpu:H200:2", "--time=3-00:00:00", "--reservation=cse_gpu"):
        assert f"#SBATCH {directive}" in source
    assert "#SBATCH --partition" not in source
