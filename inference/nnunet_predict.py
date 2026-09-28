"""Ensemble selected nnU-Net folds using final checkpoints and save probabilities."""

import argparse
from pathlib import Path
import subprocess

def run_prediction(model_dataset: int,
                   fold: tuple,
                   input_folder: Path,
                   output_folder: Path,
                   config: str,
                   trainer: str,
                   plan: str,
                   device: str = "cuda",
                ):
    if config not in ["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"]:
        raise ValueError(f"Invalid config: {config}. Must be one of '2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres'.")
    if plan not in ["nnUNetResEncUNetMPlans", "nnUNetPlans", "FineTuningPlan"]:
        raise ValueError(f"Invalid plan: {plan}. Must be one of 'nnUNetResEncUNetMPlans', 'nnUNetPlans'.")

    print(f"Running prediction with model {model_dataset}, fold {fold}, on folder {input_folder}")

    #Creation of output folder path
    identifier = trainer + "_" + config

    output_folder = output_folder / identifier
    output_folder.mkdir(parents=True, exist_ok=True)

    if isinstance(fold, str):
        fold_args = [fold]
    else:
        fold_args = list(fold)
    subprocess.run([
        "nnUNetv2_predict",
        "-i", str(input_folder),
        "-o", str(output_folder),
        "-d", str(model_dataset),
        "-c", config,
        "-p", plan,
        "-tr", trainer,
        "-chk", "checkpoint_final.pth",
        "--save_probabilities",     #option for saving softmax output
        "-f", *fold_args,
        "-device", device,
    ], check=True)

    print("Prediction finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with nnU-Net v2.")
    parser.add_argument("--dataset", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"])
    parser.add_argument("--config", default="3d_fullres")
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--plan", default="nnUNetResEncUNetMPlans")
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    args = parser.parse_args()
    run_prediction(
        args.dataset, tuple(args.folds), args.input, args.output,
        args.config, args.trainer, args.plan, args.device,
    )
