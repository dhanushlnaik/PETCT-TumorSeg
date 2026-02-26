from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from skimage import filters, morphology


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-III Lite: PET hotspot segmentation + voxel volume report"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root outputs folder containing preprocessed/ and registered/",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Optional patient ID. If omitted, process all available patients.",
    )
    parser.add_argument(
        "--threshold-method",
        choices=["otsu", "percentile"],
        default="otsu",
        help="PET thresholding strategy",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Percentile value used when --threshold-method percentile",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=80,
        help="Minimum connected component size in pixels",
    )
    parser.add_argument(
        "--hole-area",
        type=int,
        default=150,
        help="Maximum hole area (pixels) to fill",
    )
    return parser.parse_args()


def list_patient_ids(preprocessed_dir: Path, patient_id: str | None) -> list[str]:
    if patient_id:
        return [patient_id]

    ids = []
    for npz_path in sorted(preprocessed_dir.glob("*.npz")):
        if npz_path.name.endswith("_augmented.npz"):
            continue
        ids.append(npz_path.stem)
    return ids


def load_patient_data(output_root: Path, patient_id: str) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    preprocessed_file = output_root / "preprocessed" / f"{patient_id}.npz"
    registered_file = output_root / "registered" / f"{patient_id}.npz"

    if not preprocessed_file.exists():
        raise FileNotFoundError(f"Missing preprocessed file: {preprocessed_file}")
    if not registered_file.exists():
        raise FileNotFoundError(f"Missing registered file: {registered_file}")

    with np.load(preprocessed_file) as p:
        ct_volume = p["ct_volume"].astype(np.float32)

    with np.load(registered_file) as r:
        registered_pet = r["registered_pet"].astype(np.float32)
        spacing_arr = np.asarray(r["ct_spacing_mm"], dtype=np.float32).reshape(-1)
        spacing_mm = (
            float(spacing_arr[0]),
            float(spacing_arr[1]),
            float(spacing_arr[2]),
        )

    if ct_volume.shape != registered_pet.shape:
        raise ValueError(
            f"Shape mismatch for {patient_id}: CT {ct_volume.shape}, registered PET {registered_pet.shape}"
        )

    return ct_volume, registered_pet, spacing_mm


def segment_hotspots(
    pet_registered: np.ndarray,
    threshold_method: str,
    percentile: float,
    min_size: int,
    hole_area: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask_slices: list[np.ndarray] = []
    thresholds: list[float] = []

    for idx in range(pet_registered.shape[0]):
        slice_pet = pet_registered[idx]

        if threshold_method == "otsu":
            thresh = float(filters.threshold_otsu(slice_pet))
        else:
            thresh = float(np.percentile(slice_pet, percentile))

        mask = slice_pet > thresh
        mask = morphology.remove_small_objects(mask, min_size=max(1, min_size))
        mask = morphology.remove_small_holes(mask, area_threshold=max(1, hole_area))

        mask_slices.append(mask.astype(np.uint8))
        thresholds.append(thresh)

    return np.stack(mask_slices, axis=0), np.asarray(thresholds, dtype=np.float32)


def compute_volume(mask_zyx: np.ndarray, spacing_zyx_mm: tuple[float, float, float]) -> dict[str, float | int]:
    voxel_volume_mm3 = float(np.prod(np.asarray(spacing_zyx_mm, dtype=np.float64)))
    positive_voxels = int(mask_zyx.sum())
    total_volume_mm3 = positive_voxels * voxel_volume_mm3
    total_volume_ml = total_volume_mm3 / 1000.0

    return {
        "positive_voxels": positive_voxels,
        "voxel_volume_mm3": voxel_volume_mm3,
        "tumor_volume_mm3": total_volume_mm3,
        "tumor_volume_ml": total_volume_ml,
    }


def save_mask(output_root: Path, patient_id: str, mask: np.ndarray, spacing_mm: tuple[float, float, float], thresholds: np.ndarray) -> Path:
    masks_dir = output_root / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    out_path = masks_dir / f"{patient_id}_mask.npz"
    np.savez_compressed(
        out_path,
        mask=mask.astype(np.uint8),
        spacing_mm=np.asarray(spacing_mm, dtype=np.float32),
        thresholds=thresholds.astype(np.float32),
    )
    return out_path


def save_report(output_root: Path, patient_id: str, report: dict) -> Path:
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_path = reports_dir / f"{patient_id}_phase3_lite_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary_csv = reports_dir / "phase3_lite_summary.csv"
    row = {
        "patient_id": report["patient_id"],
        "slices": report["slices"],
        "height": report["height"],
        "width": report["width"],
        "threshold_method": report["threshold_method"],
        "threshold_mean": report["threshold_mean"],
        "positive_voxels": report["positive_voxels"],
        "voxel_volume_mm3": report["voxel_volume_mm3"],
        "tumor_volume_mm3": report["tumor_volume_mm3"],
        "tumor_volume_ml": report["tumor_volume_ml"],
    }

    file_exists = summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return json_path


def run_for_patient(
    output_root: Path,
    patient_id: str,
    threshold_method: str,
    percentile: float,
    min_size: int,
    hole_area: int,
) -> tuple[Path, Path, dict]:
    ct_volume, registered_pet, spacing_mm = load_patient_data(output_root, patient_id)

    mask, thresholds = segment_hotspots(
        pet_registered=registered_pet,
        threshold_method=threshold_method,
        percentile=percentile,
        min_size=min_size,
        hole_area=hole_area,
    )

    volume = compute_volume(mask, spacing_mm)

    report = {
        "patient_id": patient_id,
        "slices": int(mask.shape[0]),
        "height": int(mask.shape[1]),
        "width": int(mask.shape[2]),
        "spacing_mm_zyx": [float(v) for v in spacing_mm],
        "threshold_method": threshold_method,
        "threshold_percentile": float(percentile) if threshold_method == "percentile" else None,
        "threshold_mean": float(thresholds.mean()),
        "threshold_std": float(thresholds.std()),
        **volume,
    }

    mask_path = save_mask(output_root, patient_id, mask, spacing_mm, thresholds)
    report_path = save_report(output_root, patient_id, report)
    return mask_path, report_path, report


def main() -> None:
    args = parse_args()
    output_root = args.output_root

    patient_ids = list_patient_ids(output_root / "preprocessed", args.patient_id)
    if not patient_ids:
        raise ValueError("No patients found in outputs/preprocessed")

    for patient_id in patient_ids:
        mask_path, report_path, report = run_for_patient(
            output_root=output_root,
            patient_id=patient_id,
            threshold_method=args.threshold_method,
            percentile=args.percentile,
            min_size=args.min_size,
            hole_area=args.hole_area,
        )

        print(
            f"[{patient_id}] mask -> {mask_path} | report -> {report_path} | "
            f"tumor_volume_ml={report['tumor_volume_ml']:.3f}"
        )


if __name__ == "__main__":
    main()
