import os
import numpy as np
import pydicom
import SimpleITK as sitk
from skimage import filters, morphology


# -------------------------------------------------------
# 1) LOAD FULL DICOM SERIES (PET / CT)
# -------------------------------------------------------
def load_dicom_series(folder_path):
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".dcm")
    ]

    slices = [pydicom.dcmread(f) for f in files]

    # Sort using ImagePositionPatient Z value
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    volume = np.stack([s.pixel_array for s in slices]).astype(np.float32)

    # Extract pixel spacing + slice thickness
    px = slices[0].PixelSpacing
    slice_thickness = float(getattr(slices[0], "SliceThickness", 1.0))

    spacing = (slice_thickness, px[0], px[1])  # z, y, x spacing in mm

    return volume, spacing


# -------------------------------------------------------
# 2) NORMALIZATION
# -------------------------------------------------------
def normalize(img):
    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip(img, p1, p99)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


# -------------------------------------------------------
# 3) 3D REGISTRATION PET -> CT
# -------------------------------------------------------
def register_3d(ct_vol, pet_vol, ct_spacing, pet_spacing):
    fixed = sitk.GetImageFromArray(ct_vol.astype(np.float32))
    moving = sitk.GetImageFromArray(pet_vol.astype(np.float32))

    fixed.SetSpacing(ct_spacing[::-1])  # (x,y,z)
    moving.SetSpacing(pet_spacing[::-1])

    initial_transform = sitk.CenteredTransformInitializer(
        sitk.Cast(fixed, sitk.sitkFloat32),
        sitk.Cast(moving, sitk.sitkFloat32),
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetOptimizerAsGradientDescent(1.0, 200)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetInitialTransform(initial_transform)

    final_transform = reg.Execute(
        sitk.Cast(fixed, sitk.sitkFloat32),
        sitk.Cast(moving, sitk.sitkFloat32)
    )

    resampled = sitk.Resample(
        moving, fixed, final_transform,
        sitk.sitkLinear, 0.0, moving.GetPixelID()
    )

    registered_pet = sitk.GetArrayFromImage(resampled)
    return registered_pet


# -------------------------------------------------------
# 4) PET-BASED THRESHOLD TUMOR DETECTION (3D)
# -------------------------------------------------------
def detect_tumor_3d(pet_registered):
    mask_3d = []

    for i in range(pet_registered.shape[0]):
        slice_pet = pet_registered[i]

        thresh = filters.threshold_otsu(slice_pet)
        mask = slice_pet > thresh

        mask = morphology.remove_small_objects(mask, min_size=80)
        mask = morphology.remove_small_holes(mask, area_threshold=150)

        mask_3d.append(mask.astype(np.uint8))

    return np.stack(mask_3d)


# -------------------------------------------------------
# 5) MAIN PIPELINE
# -------------------------------------------------------
def main():

    PET_FOLDER = "PETCT fusion imaging/PET_WB/"
    CT_FOLDER  = "PETCT fusion imaging/FusionCT/"

    os.makedirs("outputs/preprocessed", exist_ok=True)
    os.makedirs("outputs/registered", exist_ok=True)
    os.makedirs("outputs/detection", exist_ok=True)

    print("\nLoading CT slices...")
    ct_vol, ct_spacing = load_dicom_series(CT_FOLDER)
    print("CT volume shape:", ct_vol.shape)

    print("\nLoading PET slices...")
    pet_vol, pet_spacing = load_dicom_series(PET_FOLDER)
    print("PET volume shape:", pet_vol.shape)

    print("\nNormalizing images...")
    ct_norm = normalize(ct_vol)
    pet_norm = normalize(pet_vol)

    np.save("outputs/preprocessed/ct.npy", ct_norm)
    np.save("outputs/preprocessed/pet.npy", pet_norm)

    print("\nRegistering PET to CT...")
    registered_pet = register_3d(ct_norm, pet_norm, ct_spacing, pet_spacing)
    np.save("outputs/registered/registered_pet.npy", registered_pet)

    print("\nDetecting tumor regions...")
    mask_3d = detect_tumor_3d(registered_pet)
    np.save("outputs/detection/tumor_mask_3d.npy", mask_3d)

    print("\nPipeline completed successfully!")
    print("Results saved in outputs/")
    print("Next: You can compute tumor volume using the 3D mask.")


if __name__ == "__main__":
    main()