from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from data_io_autopet import AutoPETStudy, find_autopet_studies, load_autopet_study_as_slices, summarize_studies
from model import build_unet, dice_coefficient, dice_loss


def combined_bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    half = tf.cast(0.5, bce.dtype)
    return half * bce + half * dice_loss(y_true, y_pred)


def split_studies(
    studies: list[AutoPETStudy],
    val_fraction: float,
    seed: int,
) -> tuple[list[AutoPETStudy], list[AutoPETStudy]]:
    if len(studies) < 2:
        raise ValueError("At least two studies are required for a train/validation split.")

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(studies))
    rng.shuffle(indices)

    val_count = max(1, int(round(len(studies) * val_fraction)))
    val_indices = set(indices[:val_count].tolist())

    train_studies = [study for idx, study in enumerate(studies) if idx not in val_indices]
    val_studies = [study for idx, study in enumerate(studies) if idx in val_indices]

    if not train_studies or not val_studies:
        raise ValueError("Invalid split produced empty train or validation set.")

    return train_studies, val_studies


def build_tensor_dataset(
    studies: list[AutoPETStudy],
    target_size: tuple[int, int],
    max_slices_per_study: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for study in studies:
        x, y, _ = load_autopet_study_as_slices(
            study=study,
            target_size=target_size,
            max_slices=max_slices_per_study,
        )
        inputs.append(x)
        targets.append(y)

    return np.concatenate(inputs, axis=0), np.concatenate(targets, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 2D U-Net on AutoPET PET/CT + SEG data")
    parser.add_argument("--autopet-root", type=Path, required=True, help="Path to AutoPET NIfTI root folder")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="Directory to store checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--target-size", type=int, nargs=2, default=(256, 256), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--max-studies", type=int, default=None, help="Optional cap on number of studies for quick runs")
    parser.add_argument(
        "--max-slices-per-study",
        type=int,
        default=128,
        help="Optional cap of slices per study to control memory usage",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    all_studies = find_autopet_studies(args.autopet_root, require_seg=True)
    if args.max_studies is not None and args.max_studies > 0:
        all_studies = all_studies[: args.max_studies]

    summary = summarize_studies(all_studies)
    print(f"Found studies: {summary['num_studies']} (patients: {summary['num_patients']})")

    train_studies, val_studies = split_studies(
        studies=all_studies,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    print(f"Train studies: {len(train_studies)} | Validation studies: {len(val_studies)}")

    x_train, y_train = build_tensor_dataset(
        studies=train_studies,
        target_size=tuple(args.target_size),
        max_slices_per_study=args.max_slices_per_study,
    )
    x_val, y_val = build_tensor_dataset(
        studies=val_studies,
        target_size=tuple(args.target_size),
        max_slices_per_study=args.max_slices_per_study,
    )

    print(f"Train tensor shape: X={x_train.shape}, y={y_train.shape}")
    print(f"Val tensor shape:   X={x_val.shape}, y={y_val.shape}")

    model = build_unet(input_shape=(args.target_size[0], args.target_size[1], 2))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=combined_bce_dice_loss,
        metrics=[
            dice_coefficient,
            tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name="iou"),
        ],
    )

    run_dir = args.output_root / "training" / "baseline_unet_2d"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "best.keras"),
            monitor="val_dice_coefficient",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coefficient",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv"), append=False),
    ]

    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(run_dir / "final.keras")

    metadata = {
        "autopet_root": str(args.autopet_root),
        "target_size": list(args.target_size),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "num_studies_total": len(all_studies),
        "num_train_studies": len(train_studies),
        "num_val_studies": len(val_studies),
        "num_train_slices": int(x_train.shape[0]),
        "num_val_slices": int(x_val.shape[0]),
    }

    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    print(f"Training complete. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
