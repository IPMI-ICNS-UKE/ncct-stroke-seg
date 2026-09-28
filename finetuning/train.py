"""Fine-tune five nnU-Net folds from their corresponding baseline checkpoints."""

import argparse
import os
import subprocess
from pathlib import Path


TRAINER = "nnUNetTrainer_freeze_configurable"
PLAN = "nnUNetResEncUNetMPlans"
CONFIGURATION = "3d_fullres"
FOLDS = tuple(map(str, range(5)))


def run_finetuning(dataset: int, pretrained_model: Path, epochs: int,
                   iterations: int = 60, learning_rate: float = 1e-3) -> None:
    if epochs < 1 or iterations < 1 or not 0 < learning_rate < float("inf"):
        raise ValueError("Epochs, iterations, and learning rate must be positive and finite")
    weights = [pretrained_model / f"fold_{fold}/checkpoint_best.pth" for fold in FOLDS]
    missing = [path for path in weights if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing baseline checkpoints: {missing}")
    env = os.environ.copy()
    env.update(STROKE_FINETUNE_EPOCHS=str(epochs),
               STROKE_FINETUNE_ITERATIONS=str(iterations),
               STROKE_FINETUNE_LR=str(learning_rate))
    for fold, checkpoint in zip(FOLDS, weights):
        subprocess.run([
            "nnUNetv2_train", str(dataset), CONFIGURATION, fold, "-device", "cuda",
            "-p", PLAN, "-tr", TRAINER, "-pretrained_weights", str(checkpoint),
        ], check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=int, required=True, help="Prepared nnU-Net dataset ID")
    parser.add_argument("--pretrained-model", type=Path, required=True,
                        help="Baseline model directory containing fold_0..fold_4")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--train-iterations", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    run_finetuning(args.dataset, args.pretrained_model, args.epochs,
                   args.train_iterations, args.learning_rate)


if __name__ == "__main__":
    main()
