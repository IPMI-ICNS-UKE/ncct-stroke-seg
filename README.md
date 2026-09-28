# Stroke CT segmentation

Code for atlas-aligned, dual-channel nnU-Net segmentation of ischemic lesions on non-contrast CT, including frozen-encoder fine-tuning and probability-threshold selection.

## Requirements

Python 3.10.18 and the verified packages in `requirements.txt`, including nnU-Net 2.6.2. Skull stripping also requires TotalSegmentator and its brain weights (see `requirements-skullstripping.txt`). Supply the CT atlas, the study images and masks, and five baseline fold checkpoints. Prepared nnU-Net datasets need their own `dataset.json`, plans, and `splits_final.json`; private-cohort data and plans are not included. Set `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` in the environment.

## Workflow

Run commands from the repository root. The examples use ISLES-like settings and placeholder dataset IDs; substitute your own prepared datasets and paths.

### 1. Preprocess CTs and masks

The pipeline registers CT and mask to a 1 × 1 × 3 mm³ atlas (affine registration; linear CT and nearest-neighbor mask interpolation), skull-strips with TotalSegmentator, clips CT to 0–80 HU, and creates the mirrored CT channel in that order. It keeps intermediate registrations and transforms in `--work`:

```bash
python -m preprocessing.pipeline --source /path/to/cases --work /path/to/work \
  --output /path/to/preprocessed --atlas /path/to/ncct_skull_3mm.nii.gz
```

Each labeled source case needs `image.nii.gz` and `mask.nii.gz`; work and output directories must be separate and empty. Input spatial headers must be valid. An explicit RPI conversion is not required. Preprocessing defaults to GPU; add `--device cpu` if needed.

### 2. Prepare nnU-Net data and baseline models

Prepare two-channel nnU-Net datasets from the preprocessed CT, mirrored CT, and mask, and generate dataset-specific plans with nnU-Net. Train the five-fold baseline with nnU-Net, or supply its `fold_0`–`fold_4` `checkpoint_best.pth` files. Keep the held-out test cases out of all training and threshold-calibration datasets.

### 3. Fine-tune the final model

Install the frozen-encoder trainer into a clean nnU-Net 2.6.2 environment once, then fine-tune five folds on your prepared training dataset:

```bash
python -m nnunet_extension.install_trainer
python -m finetuning.train --dataset 126 \
  --pretrained-model /path/to/baseline_model --epochs 30 \
  --train-iterations 60 --learning-rate 0.001
```

Each fold starts from the corresponding baseline `checkpoint_best.pth`; the encoder is frozen while the remaining weights are fine-tuned. The resulting `checkpoint_final.pth` files are used for held-out prediction. The study used 10 fine-tuning epochs for Boston and 30 for ISLES, with 60 training iterations per epoch and a learning rate of 0.001; these are arguments, not cohort-specific code defaults.

### 4. Select the probability threshold

Prepare two additional nnU-Net datasets from the fine-tuning cohort, with complementary training/validation splits and shared two-channel image/label folders. Each dataset's `splits_final.json` must repeat its split for all five folds. The search fine-tunes five models per dataset, averages their predictions on that dataset's excluded validation half, then pools both halves to choose the threshold with the highest mean Dice. Do not include held-out test cases in these inputs:

```bash
python -m finetuning.threshold_search --dataset-a 124 --dataset-b 125 \
  --images /path/to/images --labels /path/to/labels \
  --pretrained-model /path/to/baseline_model --output /path/to/calibration \
  --epochs 30 --train-iterations 60 --learning-rate 0.001
```

The default threshold grid is 0.50, 0.40, 0.30, 0.20, then 0.10–0.01 in 0.01 steps; a `threshold_evaluation.csv` is saved beneath `--output`. The study selected 0.05 for Boston and 0.03 for ISLES. The ISLES fine-tuning/test split is provided in `splits/isles2024_finetune_test.json` for reference; the code does not enforce it.

### 5. Predict held-out cases

For a labeled evaluation cohort, copy the preprocessed channels into the nnU-Net inference layout, then predict with the fine-tuned model. Inference averages five folds from `checkpoint_final.pth` and saves probability maps:

```bash
python -m inference.prepare_dataset --source /path/to/heldout_preprocessed \
  --target /path/to/nnunet_inputs --dataset-name Study --label-filename mask.nii.gz
python -m inference.nnunet_predict --dataset 126 \
  --input /path/to/nnunet_inputs/Dataset_Study/imagesVal \
  --output /path/to/predictions --trainer nnUNetTrainer_freeze_configurable
```

Inference defaults to GPU; add `--device cpu` if needed. The copied labels are for evaluation, not prediction.

### 6. Apply the threshold and evaluate

Apply the threshold chosen on the fine-tuning cohort, then evaluate the resulting masks separately:

```bash
python -m postprocessing.thresholding \
  --probabilities /path/to/predictions/nnUNetTrainer_freeze_configurable_3d_fullres \
  --output /path/to/masks --threshold 0.03
python -m evaluation.full_evaluation --predictions /path/to/masks \
  --labels /path/to/nnunet_inputs/Dataset_Study/labelsVal
```

`evaluation.nnunet_evaluate` also exposes the recorded `nnUNetv2_evaluate_simple` command. NWU is downstream of segmentation: map masks back to native space and calculate it on native, unclipped CT using the method in [Sentker et al.'s aNWU repository](https://github.com/IPMI-ICNS-UKE/aNWU).

## Repository layout

`preprocessing/` contains the ordered pipeline and individual stages; `nnunet_extension/` and `finetuning/` contain the trainer and threshold search; `inference/`, `postprocessing/`, and `evaluation/` contain the final prediction steps. The ISLES split in `splits/` is documentation only.

## Citation

If you use the segmentation workflow, cite the associated stroke CT segmentation paper. For NWU computation, cite [Sentker et al., *European Radiology* (2025)](https://doi.org/10.1007/s00330-025-12238-0) and refer to [aNWU](https://github.com/IPMI-ICNS-UKE/aNWU).

## License

Repository code, documentation, and split listing use [CC BY-NC 4.0](LICENSE), matching aNWU. nnU-Net is an external dependency under its own license; external data and weights are not covered.
