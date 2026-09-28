"""Copy paired CT/mask files and create the left-right mirrored CT channel."""

import argparse
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import shutil

def create_difference_map(SOURCE_DIR: Path, TARGET_DIR: Path):
    """Save `image_flipped.nii.gz`; the computed difference map is not written."""
    print(f"Calculating difference maps from {SOURCE_DIR} to {TARGET_DIR}")

    for patient_folder in SOURCE_DIR.iterdir():
        if patient_folder.is_dir():
            print(f"Searching through {patient_folder}")

            ct_image_path = patient_folder / "image.nii.gz"
            mask_path = patient_folder / "mask.nii.gz"

            if ct_image_path.exists() and mask_path.exists():
                print(f"Found valid ct and mask in: {patient_folder}")

                #create output folder, if necessary create parent folders
                output_dir =  TARGET_DIR / patient_folder.name
                output_dir.mkdir(parents=True, exist_ok=True)

                #copy original image and mask
                shutil.copy(ct_image_path, output_dir / "image.nii.gz")
                shutil.copy(mask_path, output_dir / "mask.nii.gz")

                #read ct and mask
                ct_image = sitk.ReadImage(ct_image_path)

                print("Flipping brain and calculating corresponding difference map!")

                ct_array = sitk.GetArrayFromImage(ct_image) #shape [Z,Y,X]

                #flip braim
                flipped_brain = np.flip(ct_array, axis = 2)

                #calculate difference map
                difference_map = ct_array - flipped_brain

                #mask all voxels to 0 (background) where original image has 0 value on one hemi and other hemi has non-zero value
                difference_mask = ((ct_array == 0) & (flipped_brain != 0)) | ((ct_array != 0) & (flipped_brain == 0))
                difference_map[difference_mask] = 0

                #save flipped brain
                flipped_brain_sitk = sitk.GetImageFromArray(flipped_brain)
                flipped_brain_sitk.SetOrigin(ct_image.GetOrigin())
                flipped_brain_sitk.SetDirection(ct_image.GetDirection())
                flipped_brain_sitk.SetSpacing(ct_image.GetSpacing())
                sitk.WriteImage(flipped_brain_sitk, output_dir / "image_flipped.nii.gz")

            else:
                print(f"No matching files found in: {patient_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the recorded flipped CT channel.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    create_difference_map(SOURCE_DIR=args.source, TARGET_DIR=args.target)
