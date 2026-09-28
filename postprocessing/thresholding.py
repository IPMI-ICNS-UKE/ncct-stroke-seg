"""Convert saved class-probability maps to masks using a fixed threshold.

Evaluation is separate; the nnU-Net mask supplies spatial metadata only.
"""

import argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk


def thresholding(SOURCE_DIR: Path, TARGET_DIR: Path, threshold: float, class_index: int, largestCC: bool = True):
    """Write binary masks, optionally retaining only the largest component."""
    for softmax_path in SOURCE_DIR.iterdir():
        if softmax_path.suffix == ".npz":
            print(f"Found softmax file {softmax_path}")
            reference_nifti = softmax_path.parent / (softmax_path.stem + ".nii.gz")


            out_folder = TARGET_DIR
            out_folder.mkdir(exist_ok=True)

            npz = np.load(softmax_path)
            softmax = npz["softmax"] if "softmax" in npz else npz["probabilities"]

            prob = softmax[class_index]    # (Z, Y, X)

            binary = (prob >= threshold).astype(np.uint8)


            ref_img = sitk.ReadImage(str(reference_nifti))
            segmentation = sitk.GetImageFromArray(binary)

            if(largestCC):
                cc = sitk.ConnectedComponent(segmentation)
                stats = sitk.LabelShapeStatisticsImageFilter()
                stats.Execute(cc)

                labels = stats.GetLabels()

                if labels:
                    largest_label = max(labels, key=lambda l: stats.GetNumberOfPixels(l))
                    segmentation = sitk.BinaryThreshold(
                        cc,
                        lowerThreshold=largest_label,
                        upperThreshold=largest_label,
                        insideValue=1,
                        outsideValue=0
                    )
                else:
                    print("No largest label!")

            segmentation.CopyInformation(ref_img)
            out_name = f"{softmax_path.stem}.nii.gz"
            out_path = out_folder / out_name
            sitk.WriteImage(segmentation, str(out_path))

            print("Finished!")
            print("Written to:", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a selected probability threshold.")
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()
    thresholding(args.probabilities, args.output, args.threshold, 1, False)
