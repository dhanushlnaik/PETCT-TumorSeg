from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import SimpleITK as sitk
from skimage.transform import resize


def load_dicom_series(folder: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load a DICOM series as a z,y,x ndarray and return volume with spacing in mm."""
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    dicom_paths = sorted([p for p in folder.iterdir() if p.suffix.lower() == ".dcm"])
    if not dicom_paths:
        raise ValueError(f"No DICOM files found in {folder}")

    slices = [pydicom.dcmread(str(path)) for path in dicom_paths]

    def sort_key(ds):
        if hasattr(ds, "ImagePositionPatient"):
            return float(ds.ImagePositionPatient[2])
        if hasattr(ds, "InstanceNumber"):
            return float(ds.InstanceNumber)
        return ds.filename

    slices.sort(key=sort_key)

    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])

    pixel_spacing = getattr(slices[0], "PixelSpacing", [1.0, 1.0])
    slice_thickness = float(getattr(slices[0], "SliceThickness", 1.0))
    spacing = (
        slice_thickness,
        float(pixel_spacing[0]),
        float(pixel_spacing[1]),
    )

    return volume, spacing


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Min-max normalize a volume to [0, 1]."""
    v_min = float(volume.min())
    v_max = float(volume.max())
    if np.isclose(v_max, v_min):
        return np.zeros_like(volume, dtype=np.float32)
    normalized = (volume - v_min) / (v_max - v_min)
    return normalized.astype(np.float32)


def resize_volume(volume: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize each slice of a volume to target_size."""
    resized = [
        resize(
            slice_,
            target_size,
            order=1,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)
        for slice_ in volume
    ]
    return np.stack(resized)


def update_spacing(
    spacing: Tuple[float, float, float],
    original_shape: Tuple[int, int, int],
    target_size: Tuple[int, int],
) -> Tuple[float, float, float]:
    """Update in-plane spacing after resizing while keeping slice thickness."""
    depth, height, width = original_shape
    target_height, target_width = target_size
    scale_y = height / target_height
    scale_x = width / target_width
    return (spacing[0], spacing[1] * scale_y, spacing[2] * scale_x)


def register_pet_to_ct(
    ct_volume: np.ndarray,
    pet_volume: np.ndarray,
    ct_spacing: Tuple[float, float, float],
    pet_spacing: Tuple[float, float, float],
) -> np.ndarray:
    """Register PET volume to CT space using SimpleITK."""
    ct_image = sitk.GetImageFromArray(ct_volume.astype(np.float32))
    pet_image = sitk.GetImageFromArray(pet_volume.astype(np.float32))

    ct_image.SetSpacing((ct_spacing[2], ct_spacing[1], ct_spacing[0]))
    pet_image.SetSpacing((pet_spacing[2], pet_spacing[1], pet_spacing[0]))

    initial_transform = sitk.CenteredTransformInitializer(
        sitk.Cast(ct_image, sitk.sitkFloat32),
        sitk.Cast(pet_image, sitk.sitkFloat32),
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration.Execute(
        sitk.Cast(ct_image, sitk.sitkFloat32),
        sitk.Cast(pet_image, sitk.sitkFloat32),
    )

    resampled = sitk.Resample(
        pet_image,
        ct_image,
        final_transform,
        sitk.sitkLinear,
        0.0,
        pet_image.GetPixelID(),
    )

    return sitk.GetArrayFromImage(resampled).astype(np.float32)


def augment_paired_volumes(
    ct_volume: np.ndarray,
    pet_volume: np.ndarray,
    augmentations_per_slice: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply identical augmentations to paired CT and PET slices."""
    if augmentations_per_slice <= 0:
        return ct_volume, pet_volume

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    rng = np.random.default_rng(seed)
    augmented_ct = []
    augmented_pet = []

    for ct_slice, pet_slice in zip(ct_volume, pet_volume):
        ct_input = ct_slice[np.newaxis, ..., np.newaxis]
        pet_input = pet_slice[np.newaxis, ..., np.newaxis]
        augmented_ct.append(ct_slice)
        augmented_pet.append(pet_slice)

        for _ in range(augmentations_per_slice):
            aug_seed = int(rng.integers(0, 1_000_000))
            ct_flow = datagen.flow(ct_input, batch_size=1, seed=aug_seed)
            pet_flow = datagen.flow(pet_input, batch_size=1, seed=aug_seed)
            augmented_ct.append(next(ct_flow)[0, :, :, 0])
            augmented_pet.append(next(pet_flow)[0, :, :, 0])

    return (
        np.stack(augmented_ct).astype(np.float32),
        np.stack(augmented_pet).astype(np.float32),
    )


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def save_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2))


def save_preview_images(
    ct_volume: np.ndarray,
    pet_volume: np.ndarray,
    registered_pet: np.ndarray,
    output_dir: Path,
    num_slices: int = 3,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    depth = ct_volume.shape[0]
    if depth == 0:
        return
    indices = np.linspace(0, depth - 1, num=min(num_slices, depth), dtype=int)

    for idx in indices:
        ct_slice = ct_volume[idx]
        pet_slice = pet_volume[idx]
        pet_reg_slice = registered_pet[idx]

        plt.imsave(output_dir / f"ct_{idx:03d}.png", ct_slice, cmap="gray")
        plt.imsave(output_dir / f"pet_{idx:03d}.png", pet_slice, cmap="magma")
        plt.imsave(output_dir / f"pet_registered_{idx:03d}.png", pet_reg_slice, cmap="magma")

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(ct_slice, cmap="gray", alpha=0.7)
        ax.imshow(pet_reg_slice, cmap="magma", alpha=0.5)
        ax.axis("off")
        fig.savefig(output_dir / f"overlay_{idx:03d}.png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def preprocess_patient(
    ct_dir: Path,
    pet_dir: Path,
    output_root: Path,
    target_size: Tuple[int, int] = (256, 256),
    augmentations_per_slice: int = 0,
    num_preview_slices: int = 3,
    patient_id: str | None = None,
) -> None:
    ct_volume, ct_spacing = load_dicom_series(ct_dir)
    pet_volume, pet_spacing = load_dicom_series(pet_dir)

    ct_norm = normalize_volume(ct_volume)
    pet_norm = normalize_volume(pet_volume)

    ct_resized = resize_volume(ct_norm, target_size)
    pet_resized = resize_volume(pet_norm, target_size)

    ct_spacing_resized = update_spacing(ct_spacing, ct_volume.shape, target_size)
    pet_spacing_resized = update_spacing(pet_spacing, pet_volume.shape, target_size)

    registered_pet = register_pet_to_ct(
        ct_resized,
        pet_resized,
        ct_spacing_resized,
        pet_spacing_resized,
    )

    aug_ct, aug_pet = augment_paired_volumes(
        ct_resized,
        registered_pet,
        augmentations_per_slice,
    )

    patient = patient_id or ct_dir.resolve().name
    preprocessed_dir = output_root / "preprocessed"
    registered_dir = output_root / "registered"
    visualization_dir = output_root / "visualizations" / patient

    save_npz(
        preprocessed_dir / f"{patient}.npz",
        ct_volume=ct_resized,
        pet_volume=pet_resized,
        ct_spacing_mm=np.array(ct_spacing_resized, dtype=np.float32),
        pet_spacing_mm=np.array(pet_spacing_resized, dtype=np.float32),
        ct_original_shape=np.array(ct_volume.shape, dtype=np.int32),
        pet_original_shape=np.array(pet_volume.shape, dtype=np.int32),
    )

    save_npz(
        registered_dir / f"{patient}.npz",
        registered_pet=registered_pet,
        ct_spacing_mm=np.array(ct_spacing_resized, dtype=np.float32),
    )

    if augmentations_per_slice > 0:
        save_npz(
            preprocessed_dir / f"{patient}_augmented.npz",
            ct_volume=aug_ct,
            pet_volume=aug_pet,
            spacing_mm=np.array(ct_spacing_resized, dtype=np.float32),
            augmentations_per_slice=np.array([augmentations_per_slice], dtype=np.int32),
        )

    metadata = {
        "patient_id": patient,
        "ct_dir": str(ct_dir),
        "pet_dir": str(pet_dir),
        "target_size": target_size,
        "ct_spacing_mm": list(ct_spacing_resized),
        "pet_spacing_mm": list(pet_spacing_resized),
        "num_slices_ct": int(ct_resized.shape[0]),
        "augmentations_per_slice": int(max(0, augmentations_per_slice)),
    }
    save_metadata(preprocessed_dir / f"{patient}_metadata.json", metadata)

    save_preview_images(
        ct_resized,
        pet_resized,
        registered_pet,
        visualization_dir,
        num_preview_slices,
    )


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess PET/CT DICOM volumes")
    parser.add_argument("--ct-dir", type=Path, default=Path("data/CT"), help="Path to CT DICOM folder")
    parser.add_argument("--pet-dir", type=Path, default=Path("data/PET"), help="Path to PET DICOM folder")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="Root directory for outputs")
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=(256, 256),
        metavar=("HEIGHT", "WIDTH"),
        help="Resize target size (height width)",
    )
    parser.add_argument(
        "--augmentations-per-slice",
        type=int,
        default=0,
        help="Number of augmented samples to generate per slice",
    )
    parser.add_argument(
        "--num-preview-slices",
        type=int,
        default=3,
        help="Number of preview slices to export",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Optional patient identifier for output files",
    )
    return parser.parse_args(args)


def main(args: Iterable[str] | None = None) -> None:
    parsed = parse_args(args)
    preprocess_patient(
        ct_dir=parsed.ct_dir,
        pet_dir=parsed.pet_dir,
        output_root=parsed.output_root,
        target_size=tuple(parsed.target_size),
        augmentations_per_slice=parsed.augmentations_per_slice,
        num_preview_slices=parsed.num_preview_slices,
        patient_id=parsed.patient_id,
    )


if __name__ == "__main__":
    main()