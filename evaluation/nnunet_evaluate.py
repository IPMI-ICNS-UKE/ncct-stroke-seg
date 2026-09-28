"""Run the recorded nnU-Net evaluation separately from prediction and thresholding."""

import argparse
import subprocess
from pathlib import Path


def run_evaluation(prediction_folder: Path, gt_folder: Path, label: str = "1") -> None:
    print(f"Starting evaluation of predictions in {prediction_folder} against ground truth in {gt_folder}")
    subprocess.run([
        "nnUNetv2_evaluate_simple",
        str(gt_folder),
        str(prediction_folder),
        "-l", label,
        "--chill",
    ], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--class-label", default="1")
    args = parser.parse_args()
    run_evaluation(args.predictions, args.labels, args.class_label)
