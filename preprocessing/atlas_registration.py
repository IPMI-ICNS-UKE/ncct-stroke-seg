"""Register CTs affinely to the atlas and apply the transform to masks."""

import argparse
from pathlib import Path
import shutil

import ants
import numpy as np


# Match the transferable FLIRT settings while retaining a full affine model.
REGISTRATION_TYPE = "Affine"
MI_BINS = 256
SAMPLING_RATE = 1.0


def atlas_registration(
    SOURCE_DIR: Path,
    TARGET_DIR: Path,
    ATLAS_PATH: Path,
    convert_masks: bool = False,
) -> None:
    """Affinely register CT images to the atlas with FLIRT-like settings."""
    print(f"Starting atlas registration of {SOURCE_DIR} to {TARGET_DIR}")
    atlas = ants.image_read(str(ATLAS_PATH))

    for patient_folder in sorted(SOURCE_DIR.iterdir()):
        if not patient_folder.is_dir():
            continue

        ct_image_path = patient_folder / "image.nii.gz"
        mask_path = patient_folder / "mask.nii.gz"
        if not ct_image_path.is_file():
            print(f"Found no valid CT in {patient_folder}")
            continue
        if convert_masks and not mask_path.is_file():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        print(f"Registering {patient_folder.name}")
        ct_image = ants.image_read(str(ct_image_path))
        registration = ants.registration(
            fixed=atlas,
            moving=ct_image,
            type_of_transform=REGISTRATION_TYPE,
            aff_metric="mattes",
            aff_sampling=MI_BINS,
            aff_random_sampling_rate=SAMPLING_RATE,
        )

        # FLIRT uses trilinear interpolation and a background value of zero.
        ct_registered = ants.apply_transforms(
            fixed=atlas,
            moving=ct_image,
            transformlist=registration["fwdtransforms"],
            interpolator="linear",
            defaultvalue=0,
        )
        ct_array = np.nan_to_num(ct_registered.numpy(), copy=False)
        ct_array[ct_array < 1] = 0
        ct_registered = ct_registered.new_image_like(ct_array)

        output_dir = TARGET_DIR / patient_folder.name
        output_dir.mkdir(parents=True, exist_ok=True)
        ants.image_write(ct_registered, str(output_dir / "image.nii.gz"))
        shutil.copy2(
            registration["fwdtransforms"][0],
            output_dir / "final_transform.mat",
        )

        if convert_masks:
            mask = ants.image_read(str(mask_path))
            mask_registered = ants.apply_transforms(
                fixed=atlas,
                moving=mask,
                transformlist=registration["fwdtransforms"],
                interpolator="nearestNeighbor",
                defaultvalue=0,
            )
            mask_array = (mask_registered.numpy() > 0.5).astype(np.uint8)
            mask_registered = mask_registered.new_image_like(mask_array).clone(
                "unsigned char"
            )
            ants.image_write(mask_registered, str(output_dir / "mask.nii.gz"))

        print(f"Registration successful and saved at {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Affine CT-to-atlas registration.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--with-masks", action="store_true")
    args = parser.parse_args()
    atlas_registration(
        SOURCE_DIR=args.source,
        TARGET_DIR=args.target,
        ATLAS_PATH=args.atlas,
        convert_masks=args.with_masks,
    )
