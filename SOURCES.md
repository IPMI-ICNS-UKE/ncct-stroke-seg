# Source mapping

The repository adapts existing files from `nnUNet_Segmentation`; originals remain untouched. Path and CLI changes do not alter the image-processing operations.

`preprocessing/pipeline.py` only sequences the four existing preprocessing functions. It clips CT in the skull-stripped directory so the paired mask remains available, and retains registration transforms in the work directory.

| Module | Existing source |
|---|---|
| `preprocessing/atlas_registration.py` | `nwu_project/preprocessing_pipeline/atlas_registration_ants.py` |
| `preprocessing/skull_stripping.py` | `nwu_project/preprocessing_pipeline/skull_stripping.py` |
| `preprocessing/hu_clipping.py` | `nwu_project/preprocessing_pipeline/hu_clipping.py` |
| `preprocessing/create_flipped_channel.py` | `nwu_project/preprocessing_pipeline/create_difference_map.py` |
| `inference/prepare_dataset.py` | `nwu_project/dataset_creation/create_inference_dataset.py` |
| `inference/nnunet_predict.py` | `nwu_project/evaluation/nnUNet_predict.py` (now explicitly requests the same default `checkpoint_final.pth`) |
| `threshold_selection/thresholding.py` | `nwu_project/postprocessing_pipeline/thresholding.py` |
| `threshold_selection/full_evaluation.py` | `nwu_project/evaluation/full_evaluation.py` (unused AISD-only output removed; scoring unchanged) |
| `finetuning/threshold_search.py` | `thresholding_full.py`, with paths/dataset IDs exposed and threshold selected by mean Dice without its repeated expected-range gate |
| `nnunet_extension/nnUNetTrainer_freeze_test.py` | `nnUNet/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_freeze_test.py` (exact copy) |

`finetuning/train.py` and `nnunet_extension/nnUNetTrainer_freeze_configurable.py` expose epochs, training iterations, and learning rate while inheriting the recorded frozen-encoder method. The copied nnU-Net trainer retains its separate Apache 2.0 license in `nnunet_extension/LICENSE.nnunet`; the repository's original material follows the root `LICENSE`.
