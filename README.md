# Stroke CT segmentation

nnU-Net segmentation of ischemic lesions on non-contrast CT, initialized with nnFoundationCNN. Original and mirrored CT share a ResEncM encoder. Local comparison modules enhance two encoder skip maps; two image-derived support channels exclude invalid comparisons. The pipeline includes source training, selective domain fine-tuning, and evaluation at a fixed probability threshold of 0.5.

## Requirements

Use Python 3.10 and a CUDA GPU supporting bfloat16 (the study used 16 GB VRAM). Create a separate environment and install the pinned nnU-Net revision and dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-skullstripping.txt
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
export nnUNet_compile=true
```

Skull stripping needs TotalSegmentator's brain weights. Supply a CT atlas at 1 × 1 × 3 mm³, CT images, and lesion annotations. Private data and trained study models are not included. Download the encoder checkpoint from the official [nnFoundationCNN repository](https://huggingface.co/MIC-DKFZ/nnFoundationCNN); it is not redistributed here:

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('MIC-DKFZ/nnFoundationCNN', 'checkpoint_final.pth', revision='8edba046f01d09771b8204b9d3e7a6c06d6e1adc', local_dir='/path/to/nnfoundation')"
```

## Workflow

Run commands from the repository root. Replace the example paths and dataset names with your own. Commands refuse to overwrite completed models or populated output directories.

### 1. Preprocess CTs and masks

Each source case is a folder containing `image.nii.gz` and, for labeled cohorts, `mask.nii.gz`. The pipeline applies affine atlas registration (linear CT, nearest-neighbor mask interpolation), TotalSegmentator skull stripping, 0–80 HU clipping, and left-right mirroring. Use an atlas whose stored x axis is left-right. Intermediate images and transforms remain in `--work`:

```bash
python -m preprocessing.pipeline --source /path/to/source_cases --work /path/to/work \
  --output /path/to/source_prepared --atlas /path/to/ncct_skull_3mm.nii.gz
```

Source, work, and output folders must be separate; work and output must be empty. Unlabeled inference cohorts are supported. Preprocessing defaults to GPU; use `--device cpu` if needed. Apply the same preprocessing to source, adaptation, and held-out cohorts. Already atlas-aligned data can enter the subsequent individual preprocessing stages directly.

### 2. Prepare source data and train

Create four channels: CT, mirrored CT, and their binary support masks (`CT > 0`, without using lesion labels). The supplied plan fixes 1 × 1 × 3 mm³ spacing, a 48 × 160 × 160 voxel patch, batch size 2, and the study's source CT normalization statistics. Both CT channels share CT normalization; support masks are not normalized.

```bash
python -m inference.prepare_dataset --source /path/to/source_prepared \
  --target "$nnUNet_raw" --dataset-name Dataset001_Stroke --training
python -m training.prepare --dataset Dataset001_Stroke
python -m training.train --dataset Dataset001_Stroke \
  --foundation /path/to/nnfoundation/checkpoint_final.pth --folds 0 1 2 3 4
python -m training.train --dataset Dataset001_Stroke \
  --foundation /path/to/nnfoundation/checkpoint_final.pth --folds all
```

For fixed internal splits, place the nnU-Net `splits_final.json` in the preprocessed dataset folder before training. Otherwise nnU-Net generates five folds. CV folds are validated after training; `fold_all` fits one model on the entire source cohort for external use. Source training uses 100 epochs × 250 updates, AdamW, encoder LR 1e-4 and decoder/comparison LR 1e-3, with 38 warmup epochs followed by polynomial decay. The official nnU-Net dynamic transfer adapts pretrained kernels to the anisotropic plan.

Models are saved under `$nnUNet_results/Dataset001_Stroke/nnUNetTrainerSymmetryComparison__SymmetryResEncMPlans__3d_fullres/fold_{0–4,all}/checkpoint_final.pth`. Add `--resume` to continue an interrupted fold from `checkpoint_latest.pth`. The custom trainer runs directly from this repository; no installed nnU-Net files are edited.

### 3. Select fine-tuning duration

Prepare a separate dataset containing **only adaptation cases**, using the same source normalization and geometry. In the commands below, `BASELINE` is the source `fold_all/checkpoint_final.pth` path:

```bash
export BASELINE=/path/to/source_model/fold_all/checkpoint_final.pth
python -m inference.prepare_dataset --source /path/to/adaptation_prepared \
  --target "$nnUNet_raw" --dataset-name Dataset002_Target --training
python -m training.prepare --dataset Dataset002_Target --source-checkpoint "$BASELINE"
python -m finetuning.prepare_splits --labels "$nnUNet_raw/Dataset002_Target/labelsTr" \
  --output "$nnUNet_preprocessed/Dataset002_Target/splits_final.json" --seed 12345
python -m finetuning.select_duration --dataset Dataset002_Target \
  --source-checkpoint "$BASELINE" --output /path/to/duration_cv --epochs 5 10 15 20
```

Two complementary folds each start from the same source model. Candidate snapshots share the same constant learning rates, so one training trajectory per fold supplies all durations. Selection uses pooled out-of-fold median Dice; ties favor shorter adaptation. Per-case scores, duration summaries, predictions, and checkpoints are saved under `--output`. Change the duration grid for the target domain. To reuse a fixed split, supply `--calibration-a-cases /path/to/case_ids.txt` instead of `--seed`.

### 4. Fine-tune on all adaptation cases

```bash
python -m finetuning.train --dataset Dataset002_Target \
  --source-checkpoint "$BASELINE" --epochs 10
```

The study settings are 10 epochs for Boston and 40 for ISLES, with 20 updates per epoch. Only comparison modules, the final decoder stage/segmentation head (LR 1e-3), and early encoder normalization affine parameters (LR 5e-4) adapt; all other parameters remain frozen. Set durations, iterations, or LRs through the corresponding command options. The final checkpoint is saved under `$nnUNet_results/Dataset002_Target/nnUNetTrainerSymmetryFineTuning__SymmetryResEncMPlans__3d_fullres/fold_all/`. Validation on this all-case refit is not used for checkpoint selection. Held-out test cases must remain outside this dataset.

The ISLES adaptation/test case lists are in `splits/isles2024_finetune_test.json`; dataset preparation does not automatically apply this listing.

### 5. Predict and evaluate

Prepare held-out CTs using step 1, create inference channels, and predict with either the source or adapted checkpoint:

```bash
python -m inference.prepare_dataset --source /path/to/heldout_prepared \
  --target /path/to/inference_inputs --dataset-name Heldout
python -m inference.nnunet_predict --checkpoint /path/to/model/checkpoint_final.pth \
  --input /path/to/inference_inputs/Heldout/imagesVal --output /path/to/predictions
python -m evaluation.full_evaluation --predictions /path/to/predictions \
  --labels /path/to/inference_inputs/Heldout/labelsVal --output /path/to/metrics
```

Inference uses Gaussian sliding-window averaging and mirror augmentation with one final model. Binary output uses probability 0.5, without threshold optimization or connected-component filtering. Add `--save-probabilities` to retain probabilities, or `--device cpu` for CPU inference. Evaluation saves per-case CSV and summary JSON, including Dice, IQR, LDR (Dice > 0), the study's mAP definition (case fraction with IoU > 0.35355), and HD95. Prediction and reference folders must contain the same cases and grids.

NWU is downstream of segmentation: map masks back to native space and calculate it on native, unclipped CT using [aNWU](https://github.com/IPMI-ICNS-UKE/aNWU), including its connected-component handling.

## Code and citation

`nnunet_extension/model.py` contains the comparison modules and shared encoder; `augmentation.py` keeps CT intensity augmentation paired and support masks intact. `trainer.py` and `optimizers.py` define source training and selective adaptation. The remaining folders follow the workflow above.

Cite the associated stroke CT segmentation paper when using this pipeline, alongside [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and [nnFoundation](https://github.com/MIC-DKFZ/nnssl). For NWU, cite [Sentker et al., *European Radiology* (2025)](https://doi.org/10.1007/s00330-025-12238-0).

## License

Repository code, documentation, and split listing use [CC BY-NC 4.0](LICENSE), matching aNWU. External software, data, and nnFoundation weights retain their own licenses.
