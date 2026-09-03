#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more directories containing STPE JSONL files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON file",
    )
    return parser.parse_args()


def quantile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return None

    probabilities = [
        0.00,
        0.01,
        0.50,
        0.90,
        0.95,
        0.99,
        0.999,
        1.00,
    ]

    result = {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }

    names = [
        "min",
        "p01",
        "p50",
        "p90",
        "p95",
        "p99",
        "p999",
        "max",
    ]

    for name, value in zip(
        names,
        np.quantile(values, probabilities),
    ):
        result[name] = float(value)

    return result


def main():
    args = parse_args()

    files = []

    for raw_path in args.paths:
        path = Path(raw_path)

        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                path.rglob("stpe_stats_rank*.jsonl")
            )

    files = sorted(set(files))

    if not files:
        raise SystemExit(
            "ERROR: no stpe_stats_rank*.jsonl files found"
        )

    records = []

    for path in files:
        for line_number, line in enumerate(
            path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines(),
            start=1,
        ):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"{path}:{line_number}: {error}"
                ) from error

    alpha_values = []
    healthy_alpha_values = []
    alpha_sample_max = []
    gamma_values = []
    scale_values = []
    denominator_values = []
    eps_fraction_values = []
    gamma_star_values = []
    f_end_values = []
    f_end_ratio_values = []
    f_step_values = []
    noise_values = []
    expected_v2_values = []
    correction_ratio_values = []

    nonfinite_alpha_count = 0
    nonfinite_f_count = 0
    monotonic_violation_count = 0

    for record in records:
        alpha_all = np.asarray(
            record["alpha"],
            dtype=np.float64,
        )

        # alpha[0] is not used in delta_f.
        alpha_used = alpha_all[1:]

        gamma_all = np.asarray(
            record["gamma"],
            dtype=np.float64,
        )
        gamma_used = gamma_all[1:]

        f = np.asarray(
            record["f"],
            dtype=np.float64,
        )

        noise = np.asarray(
            record["noise"],
            dtype=np.float64,
        )

        expected_v2 = np.asarray(
            record["expected_v2"],
            dtype=np.float64,
        )

        alpha_values.extend(alpha_used.tolist())
        gamma_values.extend(gamma_used.tolist())
        noise_values.extend(noise.tolist())
        expected_v2_values.extend(
            expected_v2.tolist()
        )

        if alpha_used.size:
            alpha_sample_max.append(
                float(np.max(alpha_used))
            )

        scale = float(record["scale"])
        denominator = float(record["denominator"])
        eps_fraction = float(record["eps_fraction"])
        gamma_star = float(record["gamma_star"])
        eps = float(record["eps"])

        scale_values.append(scale)
        denominator_values.append(denominator)
        eps_fraction_values.append(eps_fraction)
        gamma_star_values.append(gamma_star)

        if eps_fraction <= 0.1:
            healthy_alpha_values.extend(
                alpha_used.tolist()
            )

        if f.size:
            f_end = float(f[-1])
            f_end_values.append(f_end)

            raw_t_end = max(int(record["time_size"]) - 1, 1)
            f_end_ratio_values.append(
                f_end / raw_t_end
            )

        if f.size > 1:
            f_step_values.extend(np.diff(f).tolist())

            if np.any(np.diff(f) < -1.0e-7):
                monotonic_violation_count += 1

        correction_ratio = (
            2.0 * noise
            / np.maximum(expected_v2, eps)
        )
        correction_ratio_values.extend(
            correction_ratio.tolist()
        )

        nonfinite_alpha_count += int(
            record.get("alpha_nonfinite_count", 0)
        )
        nonfinite_f_count += int(
            record.get("f_nonfinite_count", 0)
        )

    alpha_array = np.asarray(
        alpha_values,
        dtype=np.float64,
    )
    gamma_array = np.asarray(
        gamma_values,
        dtype=np.float64,
    )
    scale_array = np.asarray(
        scale_values,
        dtype=np.float64,
    )
    eps_fraction_array = np.asarray(
        eps_fraction_values,
        dtype=np.float64,
    )
    gamma_star_array = np.asarray(
        gamma_star_values,
        dtype=np.float64,
    )

    eps_reference = float(records[0]["eps"])

    healthy_summary = quantile_summary(
        healthy_alpha_values
    )

    summary = {
        "files": [str(path) for path in files],
        "num_sequences": len(records),
        "num_alpha_values_used": int(
            alpha_array.size
        ),
        "window_sizes": sorted({
            int(record["window_size"])
            for record in records
        }),
        "noise_modes": sorted({
            record["noise_mode"]
            for record in records
        }),
        "time_sizes": sorted({
            int(record["time_size"])
            for record in records
        }),
        "alpha_used": quantile_summary(
            alpha_values
        ),
        "alpha_healthy_eps_fraction_le_0.1": (
            healthy_summary
        ),
        "alpha_sample_max": quantile_summary(
            alpha_sample_max
        ),
        "gamma_used": quantile_summary(
            gamma_values
        ),
        "gamma_star": quantile_summary(
            gamma_star_values
        ),
        "scale": quantile_summary(
            scale_values
        ),
        "denominator": quantile_summary(
            denominator_values
        ),
        "eps_fraction": quantile_summary(
            eps_fraction_values
        ),
        "f_end": quantile_summary(
            f_end_values
        ),
        "f_end_over_raw_t_end": quantile_summary(
            f_end_ratio_values
        ),
        "f_step": quantile_summary(
            f_step_values
        ),
        "noise": quantile_summary(
            noise_values
        ),
        "expected_v2": quantile_summary(
            expected_v2_values
        ),
        "noise_correction_ratio_2noise_over_expected_v2": (
            quantile_summary(correction_ratio_values)
        ),
        "zero_alpha_ratio": float(
            np.mean(alpha_array <= 1.0e-8)
        ),
        "alpha_gt_2_ratio": float(
            np.mean(alpha_array > 2.0)
        ),
        "alpha_gt_4_ratio": float(
            np.mean(alpha_array > 4.0)
        ),
        "alpha_gt_8_ratio": float(
            np.mean(alpha_array > 8.0)
        ),
        "alpha_gt_16_ratio": float(
            np.mean(alpha_array > 16.0)
        ),
        "negative_gamma_ratio": float(
            np.mean(gamma_array < 0.0)
        ),
        "gamma_star_nonpositive_ratio": float(
            np.mean(gamma_star_array <= 0.0)
        ),
        "scale_at_eps_floor_ratio": float(
            np.mean(
                scale_array
                <= eps_reference * 1.01
            )
        ),
        "eps_dominated_sequence_ratio": float(
            np.mean(eps_fraction_array > 0.1)
        ),
        "nonfinite_alpha_count_before_sanitization": (
            int(nonfinite_alpha_count)
        ),
        "nonfinite_f_count_before_sanitization": (
            int(nonfinite_f_count)
        ),
        "monotonic_violation_sequence_ratio": float(
            monotonic_violation_count
            / max(len(records), 1)
        ),
        "alpha_max_candidate": (
            None
            if healthy_summary is None
            else healthy_summary["p999"]
        ),
        "note": (
            "alpha_used excludes alpha[:,0]. "
            "Full test-time multi-view samples are included."
        ),
    }

    output_text = json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    )

    print(output_text)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_text(
            output_text + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
