#!/usr/bin/env python3
"""Convert DWT-SIREN training JSON reports to CSV.

Supported input formats:
1) Training manifest JSON (`manifest.json`) containing `bands` with `candidates`
2) Single-band comparison JSON (`comparison.json`) containing `candidates`
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _shape_hw(shape) -> Tuple[int, int]:
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        return shape[0], shape[1]
    return None, None


def _collect_manifest_rows(report: Dict) -> List[Dict]:
    rows: List[Dict] = []

    base = {
        "report_type": "manifest",
        "image_id": report.get("image_id"),
        "levels": report.get("levels"),
        "wavelet": report.get("wavelet"),
        "compare_configs": report.get("compare_configs"),
        "train_hf_bands": report.get("train_hf_bands"),
    }

    for band_key, band in (report.get("bands") or {}).items():
        h, w = _shape_hw(band.get("shape"))
        best_label = band.get("best_config_label")

        candidates = band.get("candidates") or []
        for rank, candidate in enumerate(candidates, start=1):
            config = candidate.get("config") or {}
            row = {
                **base,
                "band_key": band_key,
                "channel_name": band.get("channel_name"),
                "band_name": band.get("band_name"),
                "band_id": band.get("band_id"),
                "role": band.get("role"),
                "dense": band.get("dense"),
                "shape_h": h,
                "shape_w": w,
                "candidate_count": band.get("candidate_count"),
                "best_config_label": best_label,
                "best_training_psnr": band.get("best_training_psnr"),
                "candidate_rank": rank,
                "candidate_is_best": candidate.get("config_label") == best_label,
                "config_label": candidate.get("config_label"),
                "layers": config.get("layers"),
                "hidden_size": config.get("hidden_size"),
                "iterations": config.get("iterations"),
                "lr": config.get("lr"),
                "w0": config.get("w0"),
                "training_psnr": candidate.get("training_psnr"),
                "training_time_sec": candidate.get("training_time_sec"),
                "memory_peak_mb": candidate.get("memory_peak_mb"),
                "params": candidate.get("params"),
                "num_coeffs": candidate.get("num_coeffs"),
                "checkpoint_name": candidate.get("checkpoint_name"),
                "checkpoint_path": candidate.get("checkpoint_path"),
            }
            rows.append(row)

    return rows


def _collect_comparison_rows(report: Dict) -> List[Dict]:
    rows: List[Dict] = []

    h, w = _shape_hw(report.get("shape"))
    best_label = report.get("best_config_label")

    for rank, candidate in enumerate(report.get("candidates") or [], start=1):
        config = candidate.get("config") or {}
        row = {
            "report_type": "comparison",
            "image_id": None,
            "levels": None,
            "wavelet": None,
            "compare_configs": None,
            "train_hf_bands": None,
            "band_key": report.get("band_id"),
            "channel_name": report.get("channel_name"),
            "band_name": report.get("band_name"),
            "band_id": report.get("band_id"),
            "role": report.get("role"),
            "dense": report.get("dense"),
            "shape_h": h,
            "shape_w": w,
            "candidate_count": report.get("candidate_count"),
            "best_config_label": best_label,
            "best_training_psnr": report.get("best_training_psnr"),
            "candidate_rank": rank,
            "candidate_is_best": candidate.get("config_label") == best_label,
            "config_label": candidate.get("config_label"),
            "layers": config.get("layers"),
            "hidden_size": config.get("hidden_size"),
            "iterations": config.get("iterations"),
            "lr": config.get("lr"),
            "w0": config.get("w0"),
            "training_psnr": candidate.get("training_psnr"),
            "training_time_sec": candidate.get("training_time_sec"),
            "memory_peak_mb": candidate.get("memory_peak_mb"),
            "params": candidate.get("params"),
            "num_coeffs": candidate.get("num_coeffs"),
            "checkpoint_name": candidate.get("checkpoint_name"),
            "checkpoint_path": candidate.get("checkpoint_path"),
        }
        rows.append(row)

    return rows


def _detect_format(report: Dict) -> str:
    if isinstance(report.get("bands"), dict):
        return "manifest"
    if isinstance(report.get("candidates"), list):
        return "comparison"
    raise ValueError(
        "Unsupported JSON report format. Expected either manifest.json (with 'bands') "
        "or comparison.json (with 'candidates')."
    )


def _write_csv(rows: Iterable[Dict], output_path: Path) -> int:
    rows = list(rows)
    if not rows:
        raise ValueError("No rows found in input JSON report.")

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert train report JSON to CSV")
    parser.add_argument("input_json", type=Path, help="Path to manifest.json or comparison.json")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: same name as input with .csv)",
    )
    args = parser.parse_args()

    input_path = args.input_json
    output_path = args.output if args.output else input_path.with_suffix(".csv")

    with input_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    report_format = _detect_format(report)
    if report_format == "manifest":
        rows = _collect_manifest_rows(report)
    else:
        rows = _collect_comparison_rows(report)

    row_count = _write_csv(rows, output_path)
    print(f"Converted {input_path} ({report_format}) -> {output_path} with {row_count} rows")


if __name__ == "__main__":
    main()
