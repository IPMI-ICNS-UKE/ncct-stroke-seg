# Stroke CT segmentation pipeline

This repository collects the project's recorded CT preprocessing, ISLES fine-tuning, probability-threshold search, and nnU-Net v2 inference code. It preserves the existing operations and parameters; the source mapping is in [SOURCES.md](SOURCES.md).

## Workflow

1. Convert paired CT and lesion masks to RPI using the matching recorded orientation variant: Boston RAI→RPI, UKE RAI→RPI with origin reset, or ISLES LPI→RPI. The UKE and ISLES variants write in place.
2. Register both to the 3-mm NCCT atlas using ANTs affine registration. CT uses linear interpolation, zero background, and values below 1 are set to zero; masks use nearest-neighbor interpolation and remain binary.
3. Segment the brain with TotalSegmentator, erode the brain mask with radius `(0, 1, 1)`, fill holes, then apply it to CT and lesion mask.
4. Clamp CT intensities to **0–80 HU**. The original stage writes only `image.nii.gz`; run it in place if the paired mask must remain in the same case directory.
5. Create `image_flipped.nii.gz` by flipping the CT array along its X axis. Copy both CT channels and the mask into nnU-Net's `_0000`/`_0001` input format.
6. Fine-tune the five folds of datasets 036 and 037 from the corresponding Dataset023 `checkpoint_best.pth` weights, make complementary out-of-fold predictions, and select the threshold with highest mean Dice. The recorded search tries `.5, .4, .3, .2, .1, .09, .08, .07, .06, .05, .04, .03, .02, .01`; it saves a run only when the selected threshold is in `[.02, .04)`.
7. Run nnU-Net inference with five-fold prediction, saved probabilities, and the recorded evaluation helper.

## Layout

`preprocessing/` holds the image and mask stages; `finetuning/` holds the training and threshold-search orchestration; `threshold_selection/` holds probability thresholding and metrics; `inference/` holds nnU-Net input preparation and prediction; `nnunet_extension/` holds the unchanged custom trainer; `configs/` contains the verified dataset metadata and 3D nnU-Net plans for datasets 036 and 037.

## Requirements and resources

Python 3.10.18 and the package versions in `requirements.txt` were verified in `nnunet_dev`. Skull stripping requires the `TotalSegmentator` executable and its brain-model weights on a GPU; `requirements-skullstripping.txt` records version 2.11.0 from the separate `totalseg` environment. The scripts expect an external `ncct_skull_3mm.nii.gz` atlas, the original two-channel nnU-Net datasets and `splits_final.json` files, and pretrained Dataset023 fold checkpoints. These images, splits, and weights are not included.

Install the included trainer into the active nnU-Net 2.6.2 environment once with `python -m nnunet_extension.install_trainer`. It refuses to overwrite a different existing trainer. Set `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` to the corresponding nnU-Net data roots before training or inference. `NNUNET_WORKSPACE` is the parent directory of those three roots and the `nnUNet_predict` directory used by the threshold search.

## Examples

Run these commands from the repository root, substituting your data paths:

```bash
python -m preprocessing.rpi_conversion --source /path/rai_cases --target /path/rpi_cases
python -m preprocessing.atlas_registration --source /path/rpi_cases --target /path/registered --atlas /path/ncct_skull_3mm.nii.gz --with-masks
python -m preprocessing.skull_stripping --source /path/registered --target /path/stripped --with-masks
python -m preprocessing.hu_clipping --source /path/stripped --target /path/stripped
python -m preprocessing.create_flipped_channel --source /path/stripped --target /path/two_channel_cases
python -m inference.prepare_dataset --source /path/two_channel_cases --target /path/nnUNet_predict --dataset-name Study --label-filename mask.nii.gz
```

```bash
export NNUNET_WORKSPACE=/path/to/nnUNet_dev
export nnUNet_raw="$NNUNET_WORKSPACE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_WORKSPACE/nnUNet_preprocessed"
export nnUNet_results="$NNUNET_WORKSPACE/nnUNet_results"
python -m finetuning.threshold_search --iterations 1
python -m inference.nnunet_predict --dataset 36 --input /path/imagesVal --output /path/predictions --labels /path/labelsVal --trainer nnUNetTrainer_freeze_test
```

The inference helper follows the original labeled-cohort workflow and evaluates its predictions against `--labels`. The nnU-Net CLI uses its default `checkpoint_final.pth` here because the recorded helper does not pass `-chk`. The threshold-search evaluation also retains its original empty-mask and missing-case handling; these choices should be reported when using its scores. The exact raw-data construction script for datasets 036/037 was not identifiable and remains an external reproducibility dependency.
