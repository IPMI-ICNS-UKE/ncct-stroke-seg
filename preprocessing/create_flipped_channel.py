"""Copy prepared CTs and create their atlas-space left-right reflections."""

import argparse
from pathlib import Path
import shutil
import numpy as np
import SimpleITK as sitk


def create_flipped_channel(source: Path, target: Path):
    for case in sorted(source.iterdir()):
        if not case.is_dir() or not (case/'image.nii.gz').is_file():
            continue
        image = sitk.ReadImage(str(case/'image.nii.gz'))
        if not np.isclose(abs(image.GetDirection()[0]), 1):
            raise ValueError(f'Stored x axis must be left-right: {case}')
        destination = target/case.name
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(case/'image.nii.gz', destination/'image.nii.gz')
        if (case/'mask.nii.gz').exists():
            shutil.copy2(case/'mask.nii.gz', destination/'mask.nii.gz')
        reflected = sitk.GetImageFromArray(sitk.GetArrayFromImage(image)[:, :, ::-1].copy())
        reflected.CopyInformation(image)
        sitk.WriteImage(reflected, str(destination/'image_flipped.nii.gz'), True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    create_flipped_channel(args.source, args.target)


if __name__ == '__main__':
    main()
