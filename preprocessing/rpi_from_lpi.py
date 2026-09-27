"""Recorded ISLES LPI→RPI conversion; edits paired source files in place."""

import argparse
from pathlib import Path

import SimpleITK as sitk


def convert_in_place(source_dir: Path) -> None:
    for patient_folder in source_dir.iterdir():
        if patient_folder.is_dir():
            ct_path = patient_folder / "image.nii.gz"
            mask_path = patient_folder / "mask.nii.gz"

            if ct_path.exists() and mask_path.exists():
                print(f"Processing patient {patient_folder.name}")
                ct_image = sitk.ReadImage(str(ct_path))
                mask_image = sitk.ReadImage(str(mask_path))

                ct_image = sitk.Flip(ct_image, [True, False, False])
                mask_image = sitk.Flip(mask_image, [True, False, False])

                sitk.WriteImage(ct_image, str(ct_path))
                sitk.WriteImage(mask_image, str(mask_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    convert_in_place(parser.parse_args().source)
