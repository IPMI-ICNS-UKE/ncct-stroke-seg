# Stroke CT segmentation

Code for atlas-aligned, dual-channel nnU-Net segmentation of ischemic lesions on non-contrast CT, including frozen-encoder fine-tuning and probability-threshold selection. The modules adapt the existing research scripts; their provenance is listed in [SOURCES.md](SOURCES.md).

## Overview

1. Register CT and mask to a 1 × 1 × 3 mm³ atlas with ANTs affine registration (linear CT and nearest-neighbor mask interpolation), skull-strip with TotalSegmentator, and clip CT to 0–80 HU.
2. Create a left–right mirrored CT as the second input channel. Train the five-fold baseline with nnU-Net, or supply its five fold checkpoints for fine-tuning.
3. Fine-tune the five models with the encoder frozen. The study used 10 epochs for Boston and 30 for ISLES, 60 training iterations per epoch, and a learning rate of 0.001; these values are arguments, not cohort-specific code defaults.
4. Use two complementary fine-tuning-validation datasets to assess candidate thresholds. The reusable search script selects by mean Dice. For held-out inference, average the five models' probability maps from `checkpoint_final.pth` and apply a fixed threshold.

The study used fixed thresholds of 0.05 (Boston) and 0.03 (ISLES). NWU is downstream of segmentation: calculate it on native, unclipped CT after mapping masks back to native space. The NWU method and code are available separately in [Sentker et al.'s aNWU repository](https://github.com/IPMI-ICNS-UKE/aNWU).

## Structure

- `preprocessing/`: one pipeline command for registration, skull stripping, HU clipping, and the mirrored channel; individual stages remain available.
- `finetuning/`: configurable training and two-way threshold search.
- `threshold_selection/`, `inference/`: probability thresholding, evaluation, input preparation, and nnU-Net prediction.
- `nnunet_extension/`: unchanged frozen-encoder trainer plus its configurable subclass.
- `splits/isles2024_finetune_test.json`: the study's 75/74 ISLES subject split, provided for reference only; no script reads it.

## Requirements

Python 3.10.18 and the verified packages in `requirements.txt`, including nnU-Net 2.6.2. Skull stripping also needs TotalSegmentator and its brain weights (see `requirements-skullstripping.txt`). Supply the external CT atlas, prepared two-channel nnU-Net datasets with their own `dataset.json`, plans, and `splits_final.json`, and five baseline fold checkpoints. nnU-Net creates dataset-specific plans during preparation; this repository does not ship the private-cohort plans or images. Set `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` in the environment. Input CT/mask spatial headers must be valid; an explicit RPI conversion is not required before atlas registration.

## Usage

Run modules from the repository root. For labeled cases, the preprocessing entry point runs the four image stages in order, keeping intermediate registrations and transforms in `--work`:

```bash
python -m preprocessing.pipeline --source /path/to/cases --work /path/to/work \
  --output /path/to/preprocessed --atlas /path/to/ncct_skull_3mm.nii.gz
python -m inference.prepare_dataset --source /path/to/preprocessed \
  --target /path/to/nnunet_inputs --dataset-name Study --label-filename mask.nii.gz
```

The source cases must contain `image.nii.gz` and `mask.nii.gz`; work and output directories must be separate and empty. Preprocessing and inference default to GPU; use `--device cpu` for a CPU-only system. Install the custom trainer into the active nnU-Net environment once:

```bash
python -m nnunet_extension.install_trainer
```

For threshold calibration, prepare two nnU-Net datasets with complementary validation splits and a shared folder of two-channel images and labels. Then, for example:

```bash
python -m finetuning.threshold_search --dataset-a 124 --dataset-b 125 \
  --images /path/to/images --labels /path/to/labels \
  --pretrained-model /path/to/baseline_model --output /path/to/calibration \
  --epochs 30 --train-iterations 60 --learning-rate 0.001
```

To fine-tune a final model on your own prepared training dataset, use `python -m finetuning.train --dataset 126 --pretrained-model /path/to/baseline_model --epochs 30`. Keep the held-out test set out of training and validation. For inference, use `python -m inference.nnunet_predict --dataset 126 --input /path/to/images --output /path/to/predictions --labels /path/to/labels --trainer nnUNetTrainer_freeze_configurable`, followed by `python -m threshold_selection.thresholding --probabilities /path/to/predictions/nnUNetTrainer_freeze_configurable_3d_fullres --output /path/to/masks --labels /path/to/labels --threshold 0.03` (substitute your selected threshold).

## Citation

If you use the segmentation workflow, cite the associated stroke CT segmentation paper. For NWU computation, cite [Sentker et al., *European Radiology* (2025)](https://doi.org/10.1007/s00330-025-12238-0) and refer to [aNWU](https://github.com/IPMI-ICNS-UKE/aNWU).

## License

Original repository material uses [CC BY-NC 4.0](LICENSE), matching aNWU. The copied nnU-Net trainer retains its separate [Apache 2.0 license](nnunet_extension/LICENSE.nnunet); external data and weights are not covered.
