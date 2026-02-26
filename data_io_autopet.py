from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from skimage.transform import resize


@dataclass(frozen=True)
class AutoPETStudy:
    patient_id: str
    study_id: str
    study_path: Path
    suv_path: Path
    ctres_path: Path
    seg_path: Path


def _require_nibabel():
    try:
        nib = importlib.import_module("nibabel")
    except ImportError as exc:
        raise ImportError(
            "nibabel is required for AutoPET NIfTI loading. Install with: pip install nibabel"
        ) from exc
    return nib


def find_autopet_studies(root_dir: Path, require_seg: bool = True) -> list[AutoPETStudy]:
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"AutoPET root not found: {root_dir}")

    studies: list[AutoPETStudy] = []
    for patient_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        for study_dir in sorted([p for p in patient_dir.iterdir() if p.is_dir()]):
            suv = study_dir / "SUV.nii.gz"
            ctres = study_dir / "CTres.nii.gz"
            seg = study_dir / "SEG.nii.gz"

            if not suv.exists() or not ctres.exists():
                continue
            if require_seg and not seg.exists():
                continue

            studies.append(
                AutoPETStudy(
                    patient_id=patient_dir.name,
                    study_id=study_dir.name,
                    study_path=study_dir,
                    suv_path=suv,
                    ctres_path=ctres,
                    seg_path=seg,
                )
            )

    if not studies:
        raise ValueError(
            "No valid AutoPET studies found. Expected folders containing SUV.nii.gz, CTres.nii.gz, SEG.nii.gz"
        )

    return studies


def _load_nifti_xyz(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    nib = _require_nibabel()
    image = nib.load(str(path))
    arr = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    zooms = image.header.get_zooms()
    spacing_xyz = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return arr, spacing_xyz


def _xyz_to_zyx(volume_xyz: np.ndarray) -> np.ndarray:
    return np.transpose(volume_xyz, (2, 1, 0))


def _normalize_minmax(volume: np.ndarray) -> np.ndarray:
    v_min = float(np.min(volume))
    v_max = float(np.max(volume))
    if np.isclose(v_min, v_max):
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - v_min) / (v_max - v_min)).astype(np.float32)


def _resize_slices(volume_zyx: np.ndarray, target_size: tuple[int, int], order: int) -> np.ndarray:
    resized = [
        np.asarray(
            resize(
                slice_,
                target_size,
                order=order,
                mode="reflect",
                anti_aliasing=(order != 0),
                preserve_range=True,
            ),
            dtype=np.float32,
        )
        for slice_ in volume_zyx
    ]
    return np.stack(resized, axis=0)


def _spacing_xyz_to_zyx(spacing_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = spacing_xyz
    return z, y, x


def load_autopet_study_as_slices(
    study: AutoPETStudy,
    target_size: tuple[int, int] = (256, 256),
    max_slices: int | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    suv_xyz, suv_spacing_xyz = _load_nifti_xyz(study.suv_path)
    ct_xyz, ct_spacing_xyz = _load_nifti_xyz(study.ctres_path)
    seg_xyz, seg_spacing_xyz = _load_nifti_xyz(study.seg_path)

    suv_zyx = _xyz_to_zyx(suv_xyz)
    ct_zyx = _xyz_to_zyx(ct_xyz)
    seg_zyx = _xyz_to_zyx(seg_xyz)

    if suv_zyx.shape != ct_zyx.shape or suv_zyx.shape != seg_zyx.shape:
        raise ValueError(
            f"Shape mismatch in {study.study_path}: SUV {suv_zyx.shape}, CTres {ct_zyx.shape}, SEG {seg_zyx.shape}"
        )

    suv_zyx = _normalize_minmax(suv_zyx)
    ct_zyx = _normalize_minmax(ct_zyx)
    seg_zyx = (seg_zyx > 0.5).astype(np.float32)

    ct_resized = _resize_slices(ct_zyx, target_size, order=1)
    suv_resized = _resize_slices(suv_zyx, target_size, order=1)
    seg_resized = _resize_slices(seg_zyx, target_size, order=0)
    seg_resized = (seg_resized > 0.5).astype(np.float32)

    if max_slices is not None and max_slices > 0 and ct_resized.shape[0] > max_slices:
        indices = np.linspace(0, ct_resized.shape[0] - 1, max_slices, dtype=int)
        ct_resized = ct_resized[indices]
        suv_resized = suv_resized[indices]
        seg_resized = seg_resized[indices]

    x = np.stack((ct_resized, suv_resized), axis=-1).astype(np.float32)
    y = seg_resized[..., np.newaxis].astype(np.float32)

    spacing_zyx = _spacing_xyz_to_zyx(suv_spacing_xyz)

    if not np.allclose(suv_spacing_xyz, ct_spacing_xyz, atol=1e-3) or not np.allclose(
        suv_spacing_xyz, seg_spacing_xyz, atol=1e-3
    ):
        pass

    return x, y, spacing_zyx


def summarize_studies(studies: Sequence[AutoPETStudy]) -> dict[str, int]:
    unique_patients = {s.patient_id for s in studies}
    return {
        "num_studies": len(studies),
        "num_patients": len(unique_patients),
    }


def iter_studies(studies: Iterable[AutoPETStudy]):
    for study in studies:
        yield study
