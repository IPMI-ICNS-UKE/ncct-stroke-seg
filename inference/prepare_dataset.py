"""Copy paired CT channels and labels into the recorded inference layout."""

import argparse
import shutil
from pathlib import Path

# script for creating an inference dataset in the nnUNetv2 format
# this script also creates a directory for the corresponding labels
# for running the nnUNetv2_evaluate_folder command afterwards

# config of file system
parser = argparse.ArgumentParser(description="Copy paired CT channels into nnU-Net inference format.")
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--target", type=Path, required=True)
parser.add_argument("--dataset-name", required=True)
parser.add_argument("--label-filename", default="dwi_core_mask.nii.gz")
args = parser.parse_args()
SOURCE_DIR = args.source
TARGET_DIR = args.target

# optional second channel
COPY_FLIPPED_CHANNEL = True
FLIPPED_IMAGE_NAME = "image_flipped.nii.gz"

# dataset specs
dataset_name = args.dataset_name
dataset_dir_name = "Dataset" + "_" + dataset_name

# directory creation
DATASET_DIR = TARGET_DIR / dataset_dir_name
imagesVal = DATASET_DIR / "imagesVal"
labelsVal = DATASET_DIR / "labelsVal"

DATASET_DIR.mkdir(parents=True, exist_ok=True)
imagesVal.mkdir(parents=True, exist_ok=True)
labelsVal.mkdir(parents=True, exist_ok=True)

# search for training images and corresponding labels
case_count = 0

for folder in SOURCE_DIR.rglob("*"):
    if folder.is_dir():
        print(f"Searching through {folder}")

        image_raw = folder / "image.nii.gz"
        image_flipped_raw = folder / FLIPPED_IMAGE_NAME
        label_raw = folder / args.label_filename

        # check if image and label exist in current folder
        if image_raw.exists() and label_raw.exists():

            # if second channel is enabled, require flipped image as well
            if COPY_FLIPPED_CHANNEL and not image_flipped_raw.exists():
                print(f"Image and label exist, but flipped image is missing in: {folder}")
                continue

            print(f"Image and corresponding label exist in: {folder}")

            # copy image and label to target directory with changed names
            caseID = f"{case_count:03}"

            new_filename_image = f"{caseID}_{folder.name}_0000.nii.gz"
            new_filename_label = f"{caseID}_{folder.name}.nii.gz"

            targetFile_image = imagesVal / new_filename_image
            targetFile_label = labelsVal / new_filename_label

            shutil.copy2(image_raw, targetFile_image)
            shutil.copy2(label_raw, targetFile_label)

            # optional second channel: flipped image
            if COPY_FLIPPED_CHANNEL:
                new_filename_image_flipped = f"{caseID}_{folder.name}_0001.nii.gz"
                targetFile_image_flipped = imagesVal / new_filename_image_flipped

                shutil.copy2(image_flipped_raw, targetFile_image_flipped)

            print(f"Case-ID: {caseID}\nFiles copied successfully!")
            case_count += 1

        else:
            print(f"No matching files found in: {folder}")

print("Dataset creation finished!")
