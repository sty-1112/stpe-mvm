"""Validate independent evaluation and summarize official HMDB51 splits."""

import argparse
import csv
import json
import math
from pathlib import Path


MODES = ("vanilla_rope", "tad_rope", "m_rope", "video_rope", "video_rope_f")


def read_metrics(path):
    final = None
    for line in Path(path).read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # Allow an incomplete trailing line from an interrupted run.
        if isinstance(record, dict) and "Final top-1" in record and "Final Top-5" in record:
            final = {"top1": float(record["Final top-1"]), "top5": float(record["Final Top-5"])}
    if final is None or not all(math.isfinite(value) and 0 <= value <= 100 for value in final.values()):
        raise ValueError(f"Missing or invalid final merged evaluation metrics: {path}")
    if final["top1"] > final["top5"]:
        raise ValueError(f"Top-1 exceeds Top-5: {path}")
    return final


def validate_checkpoints(split_dir, mode):
    import torch

    manifest = json.loads((split_dir / "pe_args.json").read_text())
    expected = {}
    index = 0
    while index < len(manifest):
        key = manifest[index][2:]
        index += 1
        values = []
        while index < len(manifest) and not manifest[index].startswith("--"):
            values.append(manifest[index])
            index += 1
        expected[key] = values
    expected["num_frames"] = ["16"]
    expected["sampling_rate"] = ["2"]
    if expected.get("pos_mode") != [mode]:
        raise ValueError(f"PE manifest differs from requested mode: {split_dir}")

    for name, expected_epoch in (
        ("pretrain/checkpoint-4799.pth", 4799),
        ("finetune/checkpoint-49.pth", 49),
        # utils.save_model receives epoch="best" for the validation winner.
        ("finetune/checkpoint-best.pth", "best"),
    ):
        path = split_dir / name
        # Checkpoints are produced by this repository and include argparse.Namespace.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        epoch = checkpoint.get("epoch")
        if epoch != expected_epoch:
            raise ValueError(f"Incorrect final/best checkpoint epoch: {path}: {epoch}")
        if not checkpoint.get("model"):
            raise ValueError(f"Empty model state: {path}")
        args = checkpoint.get("args")
        for key, values in expected.items():
            actual = args.get(key) if isinstance(args, dict) else getattr(args, key, None)
            if key in ("rope_axis_dims",):
                equal = actual is not None and list(actual) == [int(value) for value in values]
            elif key in ("rope_theta", "tad_gamma", "temporal_spacing", "stpe_mix_beta"):
                equal = actual is not None and math.isclose(float(actual), float(values[0]), rel_tol=1e-12)
            elif key in ("rope_rotary_dim", "stpe_window_size", "num_frames", "sampling_rate"):
                equal = actual is not None and int(actual) == int(values[0])
            else:
                equal = actual == values[0]
            if not equal:
                raise ValueError(f"Checkpoint argument mismatch: {path}: {key}={actual}, expected {values}")
        del checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--splits", nargs="+", type=int, choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument("--validate-checkpoints", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    mode_dir = args.run_root / args.mode
    rows = []
    for split in args.splits:
        split_dir = mode_dir / f"split{split}"
        if args.validate_checkpoints:
            validate_checkpoints(split_dir, args.mode)
        row = {"split": split, **read_metrics(split_dir / "evaluation/log.txt")}
        rows.append(row)
        print(f"RESULT mode={args.mode} split={split} top1={row['top1']:.3f} top5={row['top5']:.3f}", flush=True)
    if args.check_only:
        return
    if sorted(args.splits) != [1, 2, 3]:
        parser.error("Writing a three-split summary requires each of splits 1,2,3 exactly once")
    mean = {key: sum(row[key] for row in rows) / 3 for key in ("top1", "top5")}
    result = {"mode": args.mode, "splits": rows, "mean": mean,
              "metric_source": "independent evaluation of validation-selected checkpoint-best.pth"}
    json_path = mode_dir / "results.json"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(json_path)
    with (mode_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "split", "top1", "top5"])
        writer.writeheader()
        writer.writerows({"mode": args.mode, **row} for row in rows)
        writer.writerow({"mode": args.mode, "split": "mean", **mean})
    print(f"MEAN mode={args.mode} top1={mean['top1']:.3f} top5={mean['top5']:.3f}", flush=True)


if __name__ == "__main__":
    main()
