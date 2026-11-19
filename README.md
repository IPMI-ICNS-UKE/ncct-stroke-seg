# Stroke CT segmentation

This repository provides an nnU-Net pipeline for segmenting ischemic lesions on non-contrast CT. It combines nnFoundationCNN initialization with explicit comparison of the two brain hemispheres.

The original CT and its mirrored version pass through a shared ResEncM encoder. Local comparison modules match nearby contralateral features and add this information to two decoder skip connections. Two image-derived support masks restrict comparison to valid image regions. The workflow covers preprocessing, source training, selective fine-tuning for a new domain, and evaluation at a fixed probability threshold of 0.5.

## Requirements

Use Python 3.10 and a CUDA GPU supporting bfloat16 (16 GB VRAM in the study). Create an environment and install the pinned dependencies:

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

Preprocessing requires TotalSegmentator's brain weights and a CT atlas at 1 × 1 × 3 mm³. Download the encoder checkpoint from the official [nnFoundationCNN repository](https://huggingface.co/MIC-DKFZ/nnFoundationCNN):

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download('MIC-DKFZ/nnFoundationCNN', 'checkpoint_final.pth',
    revision='8edba046f01d09771b8204b9d3e7a6c06d6e1adc', local_dir='/path/to/nnfoundation')
PY
```

## Workflow

Run commands from the repository root and replace example paths and dataset names with your own.

### 1. Preprocess CTs and masks

Each case folder contains `image.nii.gz` and, for labeled data, `mask.nii.gz`. Preprocessing applies affine atlas registration (linear CT and nearest-neighbor mask interpolation), skull stripping, 0–80 HU clipping, and left-right mirroring. The atlas's stored x axis must be left-right.

```bash
python -m preprocessing.pipeline --source /path/to/source_cases --work /path/to/work \
  --output /path/to/source_prepared --atlas /path/to/ncct_skull_3mm.nii.gz
```

Use separate source, work, and output folders, with work and output initially empty. Intermediate images and transforms remain in `--work`. Apply the same preprocessing to source, adaptation, and held-out data. Already atlas-aligned CTs can enter the individual subsequent stages directly. GPU processing is the default, with `--device cpu` available.

### 2. Prepare source data and train

`preprocessing.prepare_dataset` arranges the CT pair and generates two binary support channels (`CT > 0`). With `--training`, it creates the nnU-Net `imagesTr`/`labelsTr` layout. The following `training.prepare` command applies nnU-Net preprocessing. The supplied plan uses 1 × 1 × 3 mm³ spacing, a 48 × 160 × 160 voxel patch, batch size 2, and fixed source CT normalization statistics shared by both CT channels. Support masks remain unnormalized.

```bash
python -m preprocessing.prepare_dataset --source /path/to/source_prepared \
  --target "$nnUNet_raw" --dataset-name Dataset001_Stroke --training
python -m training.prepare --dataset Dataset001_Stroke
python -m training.train --dataset Dataset001_Stroke \
  --foundation /path/to/nnfoundation/checkpoint_final.pth --folds 0 1 2 3 4
python -m training.train --dataset Dataset001_Stroke \
  --foundation /path/to/nnfoundation/checkpoint_final.pth --folds all
```

Place an existing `splits_final.json` in the preprocessed dataset folder to reuse fixed CV splits. Otherwise nnU-Net generates five folds. Each CV fold is validated after training, and `fold_all` trains the final model on the full source cohort. Training uses 100 epochs × 250 updates, AdamW, encoder LR 1e-4 and decoder/comparison LR 1e-3, with 38 warmup epochs followed by polynomial decay. Official nnU-Net dynamic transfer adapts the pretrained kernels to the anisotropic plan.

Final checkpoints are saved under `$nnUNet_results/Dataset001_Stroke/nnUNetTrainerSymmetryComparison__SymmetryResEncMPlans__3d_fullres/fold_{0–4,all}/checkpoint_final.pth`. Use `--resume` to continue an interrupted fold. The trainer runs directly from this repository.

### 3. Fine-tune on adaptation cases

Prepare a dataset containing **only labeled adaptation cases**. `BASELINE` points to the source `fold_all/checkpoint_final.pth`, whose normalization and geometry are reused. The example uses the ISLES settings:

```bash
export BASELINE=/path/to/source_model/fold_all/checkpoint_final.pth
python -m preprocessing.prepare_dataset --source /path/to/adaptation_prepared \
  --target "$nnUNet_raw" --dataset-name Dataset002_Target --training
python -m training.prepare --dataset Dataset002_Target --source-checkpoint "$BASELINE"
python -m finetuning.train --dataset Dataset002_Target \
  --source-checkpoint "$BASELINE" --epochs 40 --iterations 20 \
  --main-lr 1e-3 --norm-lr 5e-4
```

Comparison modules and the final decoder stage/segmentation head adapt at `--main-lr`. Early encoder normalization affine parameters adapt at `--norm-lr`. Remaining parameters stay frozen. Epochs, updates per epoch, and both LRs are configurable. The final checkpoint is saved under `$nnUNet_results/Dataset002_Target/nnUNetTrainerSymmetryFineTuning__SymmetryResEncMPlans__3d_fullres/fold_all/`.

Keep held-out cases separate. The ISLES adaptation/test lists are provided in `splits/isles2024_finetune_test.json` and must be applied when assembling the datasets.

**Optional two-fold CV:** Use `finetuning.prepare_splits` and `finetuning.select_duration` to compare adaptation durations by pooled out-of-fold median Dice (see `--help`). This is optional when using the settings above.

### 4. Predict and evaluate

Preprocess held-out CTs as in step 1. Dataset preparation without `--training` creates `imagesVal` and optional `labelsVal` for prediction and evaluation. Then predict with the source or adapted checkpoint:

```bash
python -m preprocessing.prepare_dataset --source /path/to/heldout_prepared \
  --target /path/to/inference_inputs --dataset-name Heldout
python -m inference.nnunet_predict --checkpoint /path/to/model/checkpoint_final.pth \
  --input /path/to/inference_inputs/Heldout/imagesVal --output /path/to/predictions
python -m evaluation.full_evaluation --predictions /path/to/predictions \
  --labels /path/to/inference_inputs/Heldout/labelsVal --output /path/to/metrics
```

Inference uses one model with Gaussian sliding-window averaging, mirror augmentation, and threshold 0.5. Use `--save-probabilities` to retain probabilities or `--device cpu` for CPU inference. Evaluation saves per-case CSV and summary JSON with Dice, IQR, LDR (Dice > 0), the study's mAP definition (case fraction with IoU > 0.35355), and HD95. Predictions and references must have matching cases and grids.

For downstream NWU, map masks back to native space and calculate on native, unclipped CT using [aNWU](https://github.com/IPMI-ICNS-UKE/aNWU), including its connected-component handling.

## Code and citation

`nnunet_extension/model.py` implements the shared encoder and comparison modules. `augmentation.py` applies paired CT intensity augmentation while preserving support masks. `trainer.py` and `optimizers.py` define source training and selective adaptation.

Cite the associated stroke CT segmentation paper when using this pipeline, alongside [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and [nnFoundation](https://github.com/MIC-DKFZ/nnssl). For NWU, cite [Sentker et al., *European Radiology* (2025)](https://doi.org/10.1007/s00330-025-12238-0).

## License

Repository code, documentation, and split listing use [CC BY-NC 4.0](LICENSE). External software, data, and nnFoundation weights retain their own licenses.
