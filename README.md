# Stroke CT segmentation

Code for atlas-aligned, dual-channel nnU-Net segmentation of ischemic lesions on non-contrast CT, including frozen-encoder fine-tuning and probability-threshold selection.

## Requirements

Use Python 3.10.18 with the verified packages in `requirements.txt`, including nnU-Net 2.6.2. Skull stripping also needs TotalSegmentator and its brain weights (see `requirements-skullstripping.txt`). Supply a CT atlas, images and masks, and five baseline fold checkpoints. Set `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` in your environment. Private-cohort data, model weights, and nnU-Net plans are not included.

## Workflow

Run commands from the repository root. The examples use ISLES-like settings and placeholder dataset IDs. Replace them with your own paths and prepared datasets.

### 1. Preprocess CTs and masks

The pipeline registers CT and mask to a 1 × 1 × 3 mm³ atlas using an affine transform. It uses linear interpolation for CT and nearest-neighbor interpolation for masks. It then skull-strips with TotalSegmentator, clips CT to 0–80 HU, and creates the mirrored CT channel. Intermediate registrations and transforms remain in `--work`:

```bash
python -m preprocessing.pipeline --source /path/to/cases --work /path/to/work \
  --output /path/to/preprocessed --atlas /path/to/ncct_skull_3mm.nii.gz
```

Each labeled source case needs `image.nii.gz` and `mask.nii.gz`. The work and output directories must be separate and empty. Input spatial headers must be valid. An explicit RPI conversion is not required. Preprocessing defaults to GPU. Add `--device cpu` if needed.

### 2. Prepare nnU-Net data and baseline models

Train a five-fold baseline with nnU-Net, or supply its `fold_0`–`fold_4` `checkpoint_best.pth` files. Prepare three target-domain nnU-Net datasets from the preprocessed CT, mirrored CT, and mask. Each dataset needs its own ID, `dataset.json`, and nnU-Net plans, but the same fine-tuning cases with the same case IDs. Do not include held-out test cases.

The split helper reads exact case IDs from a `labelsTr` folder containing **only fine-tuning cases**. It writes five identical folds for each of the three datasets:

```bash
python -m finetuning.prepare_splits --labels /path/to/finetuning_only/labelsTr \
  --output /path/to/split_files --seed 12345
```

| Example dataset | Generated split | Train cases per fold | Validation cases per fold |
|---|---|---|---|
| Calibration A (124) | `calibration_a/splits_final.json` | Half A | Half B |
| Calibration B (125) | `calibration_b/splits_final.json` | Half B | Half A |
| Final model (126) | `final/splits_final.json` | All fine-tuning cases | The same fine-tuning cases |

Place each file in the matching prepared dataset's nnU-Net preprocessed directory. To reproduce a fixed calibration split, supply `--calibration-a-cases` with one exact case ID per line instead of `--seed`. The helper only writes split files. It does not copy images or generate plans.

### 3. Fine-tune calibration models and select a threshold

Install the frozen-encoder trainer into your nnU-Net 2.6.2 environment once. The calibration command fine-tunes five folds on each dataset, averages their predictions on the excluded validation half, then combines both halves to select the threshold with the highest mean Dice. Use the shared fine-tuning images and labels for `--images` and `--labels`:

```bash
python -m nnunet_extension.install_trainer
python -m finetuning.threshold_search --dataset-a 124 --dataset-b 125 \
  --images /path/to/finetuning_only/imagesTr \
  --labels /path/to/finetuning_only/labelsTr \
  --pretrained-model /path/to/baseline_model --output /path/to/calibration \
  --epochs 30 --train-iterations 60 --learning-rate 0.001
```

The study used 10 fine-tuning epochs for Boston and 30 for ISLES, with 60 training iterations per epoch and a learning rate of 0.001. Set these through the command arguments for your cohort. The default threshold grid is 0.50, 0.40, 0.30, 0.20, then 0.10–0.01 in 0.01 steps. Results are saved as `threshold_evaluation.csv` beneath `--output`. The study selected 0.05 for Boston and 0.03 for ISLES. The ISLES fine-tuning/test split is provided in `splits/isles2024_finetune_test.json` for reference. The code does not enforce it.

### 4. Fine-tune the final model

Fine-tune five folds on the final dataset, using all fine-tuning cases for training in every fold. Each fold starts from its corresponding baseline `checkpoint_best.pth`. The encoder stays frozen while the remaining weights are fine-tuned:

```bash
python -m finetuning.train --dataset 126 \
  --pretrained-model /path/to/baseline_model --epochs 30 \
  --train-iterations 60 --learning-rate 0.001
```

The final split lists the fine-tuning cases for validation too. It contains no test cases and is not used to choose a checkpoint. Use the resulting `checkpoint_final.pth` files for held-out prediction.

### 5. Predict held-out cases

For a labeled evaluation cohort, copy the preprocessed channels into the nnU-Net inference layout, then predict with the fine-tuned model. Inference averages five folds from `checkpoint_final.pth` and saves probability maps:

```bash
python -m inference.prepare_dataset --source /path/to/heldout_preprocessed \
  --target /path/to/nnunet_inputs --dataset-name Study --label-filename mask.nii.gz
python -m inference.nnunet_predict --dataset 126 \
  --input /path/to/nnunet_inputs/Dataset_Study/imagesVal \
  --output /path/to/predictions --trainer nnUNetTrainer_freeze_configurable
```

Inference defaults to GPU. Add `--device cpu` if needed. The copied labels are for evaluation, not prediction.

### 6. Apply the threshold and evaluate

Apply the threshold chosen on the fine-tuning cohort, then evaluate the resulting masks separately:

```bash
python -m postprocessing.thresholding \
  --probabilities /path/to/predictions/nnUNetTrainer_freeze_configurable_3d_fullres \
  --output /path/to/masks --threshold 0.03
python -m evaluation.full_evaluation --predictions /path/to/masks \
  --labels /path/to/nnunet_inputs/Dataset_Study/labelsVal
```

NWU is downstream of segmentation: map masks back to native space and calculate it on native, unclipped CT using the method in [Sentker et al.'s aNWU repository](https://github.com/IPMI-ICNS-UKE/aNWU).

## Repository layout

The folders follow the workflow. `preprocessing/` holds the ordered pipeline and its individual stages. `nnunet_extension/` and `finetuning/` hold the trainer and threshold search. `inference/`, `postprocessing/`, and `evaluation/` cover final prediction and scoring. The ISLES split in `splits/` is provided for reference.

## Citation

If you use the segmentation workflow, cite the associated stroke CT segmentation paper. For NWU computation, cite [Sentker et al., *European Radiology* (2025)](https://doi.org/10.1007/s00330-025-12238-0) and refer to [aNWU](https://github.com/IPMI-ICNS-UKE/aNWU).

## License

Repository code, documentation, and split listing use [CC BY-NC 4.0](LICENSE), matching aNWU. nnU-Net is an external dependency under its own license. External data and weights are not covered.
