from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence, Tuple

import numpy as np


@dataclass
class VolumeSample:
    """Represents a single patient's preprocessed data."""

    patient_id: str
    ct_volume: np.ndarray  # shape (depth, height, width)
    pet_volume: np.ndarray  # shape (depth, height, width)
    registered_pet: np.ndarray  # shape (depth, height, width)
    ct_spacing_mm: np.ndarray  # (z, y, x)
    pet_spacing_mm: np.ndarray  # (z, y, x)

    def to_multichannel(self) -> np.ndarray:
        """Stack CT and registered PET slices into a 4D tensor (depth, H, W, 2)."""
        return np.stack((self.ct_volume, self.registered_pet), axis=-1)


def load_patient(patient_file: Path, registered_file: Path | None = None) -> VolumeSample:
    """Load volumes and metadata for a single patient from .npz archives."""
    patient_id = patient_file.stem
    with np.load(patient_file) as data:
        ct_volume = data["ct_volume"].astype(np.float32)
        pet_volume = data["pet_volume"].astype(np.float32)
        ct_spacing = data["ct_spacing_mm"].astype(np.float32)
        pet_spacing = data["pet_spacing_mm"].astype(np.float32)

    if registered_file is None:
        registered_file = patient_file.parent.parent / "registered" / f"{patient_id}.npz"

    with np.load(registered_file) as reg_data:
        registered_pet = reg_data["registered_pet"].astype(np.float32)

    if registered_pet.shape != ct_volume.shape:
        raise ValueError(
            f"Shape mismatch between CT {ct_volume.shape} and registered PET {registered_pet.shape} for {patient_id}"
        )

    return VolumeSample(
        patient_id=patient_id,
        ct_volume=ct_volume,
        pet_volume=pet_volume,
        registered_pet=registered_pet,
        ct_spacing_mm=ct_spacing,
        pet_spacing_mm=pet_spacing,
    )


def iter_patients(preprocessed_dir: Path, registered_dir: Path | None = None) -> Iterator[VolumeSample]:
    """Yield VolumeSample instances for each patient found in the preprocessed directory."""
    preprocessed_dir = Path(preprocessed_dir)
    registered_dir = Path(registered_dir) if registered_dir else preprocessed_dir.parent / "registered"

    for npz_path in sorted(preprocessed_dir.glob("*.npz")):
        if npz_path.name.endswith("_augmented.npz"):
            continue
        patient_id = npz_path.stem
        registered_path = registered_dir / f"{patient_id}.npz"
        if not registered_path.exists():
            raise FileNotFoundError(f"Missing registered file for patient {patient_id}: {registered_path}")
        yield load_patient(npz_path, registered_path)


def train_val_split(patients: Sequence[VolumeSample], val_fraction: float = 0.2, seed: int = 42) -> Tuple[list[VolumeSample], list[VolumeSample]]:
    """Split a list of patients into train and validation lists."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(patients))
    rng.shuffle(indices)

    val_size = max(1, int(round(len(patients) * val_fraction)))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train = [patients[i] for i in train_indices]
    val = [patients[i] for i in val_indices]

    return train, val


def stack_slices(volume_samples: Sequence[VolumeSample], modality: str = "ct") -> np.ndarray:
    """Stack all slices from the selected modality across patients.

    modality options:
        - "ct": returns CT volume slices
        - "pet": returns resized PET volume slices
        - "registered_pet": returns registered PET slices
        - "multichannel": returns stacked CT + registered PET channels
    """

    modality = modality.lower()
    slices = []

    for sample in volume_samples:
        if modality == "ct":
            data = sample.ct_volume
        elif modality == "pet":
            data = sample.pet_volume
        elif modality == "registered_pet":
            data = sample.registered_pet
        elif modality == "multichannel":
            data = sample.to_multichannel()
        else:
            raise ValueError(f"Unknown modality '{modality}'")

        slices.append(data)

    return np.concatenate(slices, axis=0)
