"""Use TotalSegmentator's brain mask to strip CTs and paired lesion masks."""

import argparse
import subprocess
import SimpleITK as sitk
from pathlib import Path
import torch

#function for skull stripping of a dataset
def skull_stripping(SOURCE_DIR: Path, TARGET_DIR: Path, convert_masks: bool = False, device: str = "gpu"):
    print(f"Starting skull stripping of{SOURCE_DIR} to {TARGET_DIR}")

    for patient_folder in SOURCE_DIR.iterdir():
        if patient_folder.is_dir():
            print(f"Searching through {patient_folder}")
            ct_image_path = patient_folder / "image.nii.gz"

            #check if CT image exists
            if ct_image_path.exists():
                print(f"Found valid CT in: {patient_folder}")

                #create output folder and totalSeg folder, if necessary create parent folders
                output_dir =  TARGET_DIR / patient_folder.name
                output_dir.mkdir(parents=True, exist_ok=True)

                totalSeg_folder = output_dir / "total_seg"
                totalSeg_folder.mkdir(parents=True, exist_ok=True)

                if torch.cuda.is_available():
                    print("CUDA is available!")

                #brain segmentation with TotalSegmentator
                print(f"Running brain segmentation on {ct_image_path}")
                subprocess.run([
                    "TotalSegmentator",
                    "-i", ct_image_path,
                    "-o", totalSeg_folder,
                    "--device", device,
                    "--roi_subset_robust", "brain",
                    "-rc"
                ], check=True)

                #apply segmentation to ct
                print(f"Applying segmentation on {ct_image_path}")
                ct_image = sitk.ReadImage(ct_image_path)
                segmentation = sitk.ReadImage(totalSeg_folder / "brain.nii.gz")

                #remove remaining bone
                f_erosion = sitk.BinaryErodeImageFilter()
                f_erosion.SetKernelRadius((0,1,1))
                segmentation = f_erosion.Execute(segmentation)

                #fill holes
                segmentation = sitk.BinaryFillhole(segmentation)

                brain_only = sitk.Mask(ct_image, segmentation)

                #write image
                sitk.WriteImage(brain_only, output_dir / "image.nii.gz")

                if(convert_masks):
                    print("Converting corresponding mask!")
                    mask = sitk.ReadImage(patient_folder / "mask.nii.gz")
                    new_mask = sitk.Mask(mask, segmentation)

                    #write mask
                    sitk.WriteImage(new_mask, output_dir / "mask.nii.gz")
            else:
                print(f"No matching files found in: {patient_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skull strip atlas-aligned CTs.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--with-masks", action="store_true")
    parser.add_argument("--device", choices=("gpu", "cpu", "mps"), default="gpu")
    args = parser.parse_args()
    skull_stripping(args.source, args.target, convert_masks=args.with_masks, device=args.device)
