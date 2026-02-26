#!/usr/bin/env python3
"""
Demo Pipeline: End-to-End Tumor Segmentation & Volume Calculation

Shows the complete workflow:
  1. Load preprocessed PET/CT data
  2. Generate synthetic tumor masks
  3. Calculate tumor volumes
  4. Create comprehensive visualizations
  5. Generate polished report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters, morphology

# ============================================================================
# SECTION 1: Load Preprocessed Data
# ============================================================================

def load_preprocessed_data(
    preprocessed_dir: Path,
    registered_dir: Path,
    patient_id: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, tuple, tuple]:
    """Load CT, PET, and registered PET volumes with spacing."""
    
    with np.load(preprocessed_dir / f"{patient_id}.npz") as data:
        ct_vol = data["ct_volume"].astype(np.float32)
        pet_vol = data["pet_volume"].astype(np.float32)
        ct_spacing = tuple(data["ct_spacing_mm"])
        pet_spacing = tuple(data["pet_spacing_mm"])
    
    with np.load(registered_dir / f"{patient_id}.npz") as data:
        reg_pet_vol = data["registered_pet"].astype(np.float32)
    
    return ct_vol, pet_vol, reg_pet_vol, ct_spacing, pet_spacing


# ============================================================================
# SECTION 2: Generate Synthetic Tumor Masks
# ============================================================================

def generate_synthetic_mask(
    pet_volume: np.ndarray,
    threshold_method: str = "otsu",
    percentile: float = 95.0,
    min_size: int = 80,
    hole_area: int = 150,
) -> np.ndarray:
    """Generate binary tumor mask from PET using thresholding."""
    
    mask_3d = []
    for slice_idx, pet_slice in enumerate(pet_volume):
        
        if threshold_method == "otsu":
            thresh = filters.threshold_otsu(pet_slice)
        elif threshold_method == "percentile":
            thresh = np.percentile(pet_slice, percentile)
        else:
            raise ValueError(f"Unknown threshold method: {threshold_method}")
        
        mask = pet_slice > thresh
        mask = morphology.remove_small_objects(mask, min_size=min_size)
        mask = morphology.remove_small_holes(mask, area_threshold=hole_area)
        
        mask_3d.append(mask.astype(np.uint8))
    
    return np.stack(mask_3d)


# ============================================================================
# SECTION 3: Calculate Tumor Volumes
# ============================================================================

def calculate_volumes(
    mask_ct: np.ndarray,
    mask_pet: np.ndarray,
    spacing_mm: tuple,
) -> dict:
    """Calculate GTV (from CT) and MTV (from PET) in mm³ and mL."""
    
    z_spacing, y_spacing, x_spacing = spacing_mm
    voxel_volume_mm3 = z_spacing * y_spacing * x_spacing
    
    gtv_voxels = np.sum(mask_ct)
    mtv_voxels = np.sum(mask_pet)
    
    gtv_mm3 = gtv_voxels * voxel_volume_mm3
    mtv_mm3 = mtv_voxels * voxel_volume_mm3
    
    gtv_ml = gtv_mm3 / 1000.0
    mtv_ml = mtv_mm3 / 1000.0
    
    return {
        "gtv_voxels": int(gtv_voxels),
        "mtv_voxels": int(mtv_voxels),
        "voxel_volume_mm3": float(voxel_volume_mm3),
        "gtv_mm3": float(gtv_mm3),
        "mtv_mm3": float(mtv_mm3),
        "gtv_ml": float(gtv_ml),
        "mtv_ml": float(mtv_ml),
    }


# ============================================================================
# SECTION 4: Create Enhanced Visualizations
# ============================================================================

def create_comprehensive_visualizations(
    ct_vol: np.ndarray,
    pet_vol: np.ndarray,
    reg_pet_vol: np.ndarray,
    mask_ct: np.ndarray,
    mask_pet: np.ndarray,
    patient_id: str,
    output_dir: Path,
    num_slices: int = 10,
) -> None:
    """Create enhanced multi-slice visualizations with masks (optimized)."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    depth = ct_vol.shape[0]
    if num_slices > depth:
        num_slices = depth
    
    indices = np.linspace(0, depth - 1, num=num_slices, dtype=int)
    
    for idx in indices:
        # Single compact visualization per slice: overlay with masks
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # CT + CT mask
        ax = axes[0]
        ax.imshow(ct_vol[idx], cmap="gray")
        ax.contour(mask_ct[idx], colors="red", linewidths=1.5, levels=[0.5])
        ax.set_title(f"CT + GTV (Slice {idx})", fontsize=10)
        ax.axis("off")
        
        # PET + PET mask
        ax = axes[1]
        ax.imshow(reg_pet_vol[idx], cmap="magma")
        ax.contour(mask_pet[idx], colors="cyan", linewidths=1.5, levels=[0.5])
        ax.set_title(f"PET + MTV (Slice {idx})", fontsize=10)
        ax.axis("off")
        
        plt.tight_layout()
        plt.savefig(output_dir / f"slice_{idx:03d}_masks.png", dpi=80, bbox_inches="tight")
        plt.close()
    
    print(f"✓ Created {num_slices} visualization images in {output_dir}")


# ============================================================================
# SECTION 5: Generate Demo Report
# ============================================================================

def generate_demo_report(
    patient_id: str,
    ct_vol: np.ndarray,
    reg_pet_vol: np.ndarray,
    mask_ct: np.ndarray,
    mask_pet: np.ndarray,
    spacing_mm: tuple,
    volumes: dict,
    output_dir: Path,
) -> dict:
    """Generate a comprehensive demo report."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    z_spacing, y_spacing, x_spacing = spacing_mm
    num_slices = ct_vol.shape[0]
    slices_with_tumor_ct = np.sum(np.sum(mask_ct, axis=(1, 2)) > 0)
    slices_with_tumor_pet = np.sum(np.sum(mask_pet, axis=(1, 2)) > 0)
    
    report = {
        "patient_id": patient_id,
        "pipeline_stage": "demo",
        "timestamp": str(np.datetime64("today")),
        
        # Volume Dimensions
        "volume_info": {
            "num_slices_z": int(num_slices),
            "height_y": int(ct_vol.shape[1]),
            "width_x": int(ct_vol.shape[2]),
            "spacing_mm_zyx": [float(z_spacing), float(y_spacing), float(x_spacing)],
        },
        
        # Tumor Metrics
        "tumor_metrics": {
            "gtv_voxels": volumes["gtv_voxels"],
            "mtv_voxels": volumes["mtv_voxels"],
            "slices_with_tumor_ct": int(slices_with_tumor_ct),
            "slices_with_tumor_pet": int(slices_with_tumor_pet),
            "gtv_mm3": volumes["gtv_mm3"],
            "mtv_mm3": volumes["mtv_mm3"],
            "gtv_ml": volumes["gtv_ml"],
            "mtv_ml": volumes["mtv_ml"],
        },
        
        # Intensity Statistics
        "intensity_stats": {
            "ct_min": float(ct_vol.min()),
            "ct_max": float(ct_vol.max()),
            "ct_mean": float(ct_vol.mean()),
            "ct_std": float(ct_vol.std()),
            "pet_min": float(reg_pet_vol.min()),
            "pet_max": float(reg_pet_vol.max()),
            "pet_mean": float(reg_pet_vol.mean()),
            "pet_std": float(reg_pet_vol.std()),
        },
        
        # Summary
        "summary": f"Patient {patient_id}: GTV={volumes['gtv_ml']:.2f} mL, MTV={volumes['mtv_ml']:.2f} mL",
    }
    
    # Save JSON report
    report_path = output_dir / f"{patient_id}_demo_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return report


# ============================================================================
# MAIN DEMO PIPELINE
# ============================================================================

def main(args: argparse.Namespace) -> None:
    """Run the complete demo pipeline."""
    
    print("\n" + "="*70)
    print("  PET/CT TUMOR SEGMENTATION DEMO PIPELINE")
    print("="*70)
    
    # Paths
    preprocessed_dir = Path(args.output_root) / "preprocessed"
    registered_dir = Path(args.output_root) / "registered"
    masks_dir = Path(args.output_root) / "masks"
    vis_dir = Path(args.output_root) / "visualizations"
    reports_dir = Path(args.output_root) / "reports"
    
    masks_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[1] Loading preprocessed data for patient '{args.patient_id}'...")
    ct_vol, pet_vol, reg_pet_vol, ct_spacing, pet_spacing = load_preprocessed_data(
        preprocessed_dir, registered_dir, args.patient_id
    )
    print(f"    ✓ CT shape: {ct_vol.shape}, PET shape: {pet_vol.shape}")
    print(f"    ✓ CT spacing (mm): {ct_spacing}")
    
    print(f"\n[2] Generating synthetic tumor masks...")
    mask_ct = generate_synthetic_mask(ct_vol, threshold_method="otsu")
    mask_pet = generate_synthetic_mask(reg_pet_vol, threshold_method="otsu")
    print(f"    ✓ CT mask shape: {mask_ct.shape}, positive voxels: {np.sum(mask_ct)}")
    print(f"    ✓ PET mask shape: {mask_pet.shape}, positive voxels: {np.sum(mask_pet)}")
    
    # Save masks
    np.savez_compressed(
        masks_dir / f"{args.patient_id}_masks.npz",
        ct_mask=mask_ct,
        pet_mask=mask_pet,
        spacing_mm=np.array(ct_spacing),
    )
    print(f"    ✓ Masks saved to {masks_dir}")
    
    print(f"\n[3] Calculating tumor volumes...")
    volumes = calculate_volumes(mask_ct, mask_pet, ct_spacing)
    print(f"    ✓ GTV (Gross Tumor Volume):     {volumes['gtv_ml']:.2f} mL")
    print(f"    ✓ MTV (Metabolic Tumor Volume): {volumes['mtv_ml']:.2f} mL")
    
    print(f"\n[4] Creating comprehensive visualizations...")
    patient_vis_dir = vis_dir / args.patient_id / "masks"
    create_comprehensive_visualizations(
        ct_vol, pet_vol, reg_pet_vol, mask_ct, mask_pet,
        args.patient_id,
        patient_vis_dir,
        num_slices=args.num_slices,
    )
    
    print(f"\n[5] Generating demo report...")
    report = generate_demo_report(
        args.patient_id,
        ct_vol, reg_pet_vol, mask_ct, mask_pet,
        ct_spacing,
        volumes,
        reports_dir,
    )
    print(f"    ✓ Report saved to {reports_dir / f'{args.patient_id}_demo_report.json'}")
    
    print(f"\n" + "="*70)
    print(f"  DEMO PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\n📊 SUMMARY FOR PRESENTATION:")
    print(f"   Patient ID:    {args.patient_id}")
    print(f"   GTV (CT-based): {volumes['gtv_ml']:.1f} mL")
    print(f"   MTV (PET-based): {volumes['mtv_ml']:.1f} mL")
    print(f"\n📁 Output Locations:")
    print(f"   • Masks:         {masks_dir}")
    print(f"   • Visualizations: {patient_vis_dir}")
    print(f"   • Reports:       {reports_dir}")
    print(f"\n✓ Ready for presentation tomorrow!\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo Pipeline: End-to-End Tumor Segmentation & Volume Calculation"
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default="demo_patient",
        help="Patient ID to process",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root output directory",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=10,
        help="Number of slices to visualize",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
