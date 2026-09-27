import argparse
import SimpleITK as sitk
from pathlib import Path

#script for fixing direction of Boston Dataset from RAI to RPI
def fix_direction_Boston(SOURCE_DIR: Path, TARGET_DIR: Path):
    print(f"Starting fixing of direction of {SOURCE_DIR} to {TARGET_DIR}")

    for patient_folder in SOURCE_DIR.iterdir():
        if patient_folder.is_dir():
            print(f"Searching through {patient_folder}")
            ct_image_path = patient_folder / "image.nii.gz"
            mask_path = patient_folder / "mask.nii.gz"

            #check if CT folder exists and contains data
            if ct_image_path.exists() and mask_path.exists():
                print(f"Found valid CT and mask in: {patient_folder}")

                #create output folder, if necessary create parent folders
                output_dir =  TARGET_DIR / patient_folder.name
                output_dir.mkdir(parents=True, exist_ok=True)

                #read image and mask
                ct_image = sitk.ReadImage(ct_image_path)
                mask = sitk.ReadImage(mask_path)

                #fix direction (currently RAI, desired RPI)
                ct_image_RPI = sitk.Flip(ct_image, [False, True, False])
                mask_RPI = sitk.Flip(mask, [False, True, False])



                #write corrected ct and mask
                print(f"Writing direction corrected ct and mask to {output_dir}")
                sitk.WriteImage(ct_image_RPI, output_dir / "image.nii.gz")
                sitk.WriteImage(mask_RPI, output_dir / "mask.nii.gz")

            else:
                print(f"No matching files found in: {patient_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flip paired RAI CT and mask to RPI.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    fix_direction_Boston(SOURCE_DIR=args.source, TARGET_DIR=args.target)
