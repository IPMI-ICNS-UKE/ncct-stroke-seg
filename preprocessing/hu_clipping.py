"""Clamp skull-stripped CT intensities to the configured HU interval."""

import argparse
import SimpleITK as sitk
from pathlib import Path

def hu_clipping(SOURCE_DIR: Path, TARGET_DIR: Path, range_lower: int = 0, range_upper: int = 100):
    print(f"Starting hu clipping of {SOURCE_DIR} to {TARGET_DIR}")

    for patient_folder in SOURCE_DIR.iterdir():
        if patient_folder.is_dir():
            print(f"Searching through {patient_folder}")
            ct_image_path = patient_folder / "image.nii.gz"

            #check if CT folder exists and contains data
            if ct_image_path.exists():
                print(f"Found valid CT in: {patient_folder}")

                #create output folder, if necessary create parent folders
                output_dir =  TARGET_DIR / patient_folder.name
                output_dir.mkdir(parents=True, exist_ok=True)

                #read image
                ct_image = sitk.ReadImage(str(ct_image_path))

                #clip image
                ct_image_clipped = sitk.Clamp(ct_image, lowerBound=range_lower, upperBound=range_upper)


                #write clipped ct
                print(f"Writing clipped ct to {output_dir}")
                sitk.WriteImage(ct_image_clipped, str(output_dir / "image.nii.gz"))


            else:
                print(f"No matching files found in: {patient_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clamp CT to the recorded 0–80 HU range.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    hu_clipping(args.source, args.target, range_lower=0, range_upper=80)
