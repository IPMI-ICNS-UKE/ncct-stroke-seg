# Source mapping and retained behavior

Script paths below are relative to the existing `nnUNet_Segmentation` project; configuration paths are relative to `nnUNet_project`. Originals remain untouched.

| New file | Original source | Organizational edit |
|---|---|---|
| `preprocessing/rpi_conversion.py` | `nwu_project/preprocessing_pipeline/fix_direction_Boston.py` | CLI paths instead of fixed paths. Assumes RAI input; no orientation detection or origin reset was added. |
| `preprocessing/rpi_from_rai_reset_origin.py` | `nwu_project/utils/RAI_to_RPI.py` | Extracted its in-place Y flip and subsequent zero-origin assignment into a CLI function. |
| `preprocessing/rpi_from_lpi.py` | `nwu_project/utils/LPI_to_RPI.py` | Extracted its in-place X flip into a CLI function. |
| `preprocessing/atlas_registration.py` | `nwu_project/preprocessing_pipeline/atlas_registration_ants.py` | CLI paths and mask switch; registration and resampling body unchanged. |
| `preprocessing/skull_stripping.py` | `nwu_project/preprocessing_pipeline/skull_stripping.py` | CLI paths and mask switch. |
| `preprocessing/hu_clipping.py` | `nwu_project/preprocessing_pipeline/hu_clipping.py` | CLI paths; retains the operational 0–80 HU call. Function's historical 0–100 defaults remain. |
| `preprocessing/create_flipped_channel.py` | `nwu_project/preprocessing_pipeline/create_difference_map.py` | CLI paths; the source computes but does not save a difference map. |
| `inference/prepare_dataset.py` | `nwu_project/dataset_creation/create_inference_dataset.py` | CLI paths, dataset name, and label filename. The original label filename remains the default. |
| `inference/nnunet_predict.py` | `nwu_project/evaluation/nnUNet_predict.py` | CLI arguments; prediction and evaluation functions unchanged. |
| `finetuning/threshold_search.py` | `thresholding_full.py` | Local imports and `NNUNET_WORKSPACE` replace the fixed project root. |
| `threshold_selection/thresholding.py` | `nwu_project/postprocessing_pipeline/thresholding.py` | CLI arguments; threshold and evaluation functions unchanged. |
| `threshold_selection/full_evaluation.py` | `nwu_project/evaluation/full_evaluation.py` | CLI arguments; metric functions unchanged. |
| `nnunet_extension/nnUNetTrainer_freeze_test.py` | `nnUNet/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_freeze_test.py` | Exact file copy. |
| `configs/Dataset036*`, `configs/Dataset037*` | `nnUNet_dev/nnUNet_preprocessed` dataset JSON and `nnUNetResEncUNetMPlans.json` | Exact copies of non-image configuration files. |

The README example uses the Boston RAI→RPI variant. Choose the UKE or ISLES variant only when its recorded input orientation and origin handling apply; no automatic orientation detection was added. Existing 5-mm AISD registrations and rigid FLIRT outputs are separate experiments; this repository records the requested 3-mm ANTs affine path. The selected threshold workflow is the ISLES complementary-validation experiment. Other experimental fine-tuning scripts have different datasets and trainers and were not merged into it.

The project contains the prepared datasets 036/037 and their split files, but no identifiable script that reconstructs those exact training datasets and complementary splits from raw patient images. They remain external inputs. The threshold-search script also selects a run by mean Dice and its expected threshold interval; this repository does not claim that its saved checkpoint is the final paper model.

`nnunet_extension/LICENSE.nnunet` preserves the Apache 2.0 license accompanying the copied nnU-Net trainer source; it does not set a license for the other project scripts.
