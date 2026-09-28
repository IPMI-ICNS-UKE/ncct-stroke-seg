"""Two-way fine-tuning validation and mean-Dice probability-threshold search."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from finetuning.train import CONFIGURATION, FOLDS, PLAN, TRAINER, run_finetuning
from inference import nnunet_predict
from evaluation import full_evaluation
from postprocessing import thresholding


THRESHOLDS = (.5, .4, .3, .2, .1, .09, .08, .07, .06, .05, .04, .03, .02, .01)


def dataset_path(preprocessed: Path, dataset: int) -> Path:
    matches = list(preprocessed.glob(f"Dataset{dataset:03d}_*"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one preprocessed directory for dataset {dataset}: {matches}")
    return matches[0]


def validation_cases(preprocessed: Path, datasets: tuple[int, int]):
    splits = []
    for dataset in datasets:
        path = dataset_path(preprocessed, dataset) / "splits_final.json"
        with path.open() as file:
            folds = json.load(file)
        if len(folds) != len(FOLDS) or any(fold != folds[0] for fold in folds):
            raise RuntimeError(f"Expected five identical validation splits in {path}")
        splits.append(folds[0])
    first, second = splits
    if (set(first["train"]) != set(second["val"])
            or set(first["val"]) != set(second["train"])):
        raise RuntimeError("The two validation datasets must have complementary splits")
    return tuple(map(sorted, (first["val"], second["val"])))


def link_exact(sources, destination: Path, suffixes=(".nii.gz", ".npz")):
    """Link only expected files, without replacing existing data."""
    destination.mkdir(parents=True, exist_ok=True)
    wanted = {source.name: source.resolve() for source in sources}
    present = {p.name for p in destination.iterdir() if p.name.endswith(suffixes)}
    if extra := present - set(wanted):
        raise RuntimeError(f"Unexpected files in {destination}: {sorted(extra)}")
    for name, source in wanted.items():
        target = destination / name
        if target.is_symlink() and target.resolve() == source:
            continue
        if target.exists() or target.is_symlink():
            raise RuntimeError(f"Refusing to replace {target}")
        target.symlink_to(source)


def prepare_inputs(case_groups, images: Path, labels: Path, inputs):
    for cases, destination in zip(case_groups, inputs):
        sources = []
        for case in cases:
            modalities = sorted(images.glob(f"{case}_*.nii.gz"))
            if len(modalities) != 2 or not (labels / f"{case}.nii.gz").is_file():
                raise RuntimeError(f"Incomplete two-channel image or label for {case}")
            sources.extend(modalities)
        link_exact(sources, destination, (".nii.gz",))


def predict(datasets, case_groups, inputs, outputs, oof: Path):
    oof_sources = []
    for dataset, cases, input_folder, output in zip(datasets, case_groups, inputs, outputs):
        nnunet_predict.run_prediction(
            dataset, FOLDS, input_folder, output,
            CONFIGURATION, TRAINER, PLAN,
        )
        folder = output / f"{TRAINER}_{CONFIGURATION}"
        required = [folder / f"{case}{suffix}" for case in cases
                    for suffix in (".nii.gz", ".npz")]
        if any(not path.is_file() for path in required):
            raise RuntimeError(f"Missing validation predictions in {folder}")
        found = {p.name for p in folder.iterdir() if p.name.endswith((".nii.gz", ".npz"))}
        if found != {p.name for p in required}:
            raise RuntimeError(f"Prediction set mismatch in {folder}")
        oof_sources.extend(required)
    link_exact(oof_sources, oof)


def optimize_threshold(oof: Path, masks: Path, labels: Path, thresholds):
    rows = []
    for value in thresholds:
        thresholding.thresholding(oof, masks, value, 1, False)
        metrics = full_evaluation.full_evaluation(masks, labels)
        if metrics is None:
            raise RuntimeError(f"No cases evaluated at threshold {value}")
        mean_dice, median_dice, hd_accuracy, mean_fp, mean_ap = metrics
        rows.append(dict(threshold=value, mean_dice=mean_dice, median_dice=median_dice,
                         hd_accuracy=hd_accuracy, mean_fp=mean_fp, mAP=mean_ap))
    table = pd.DataFrame(rows)
    table.to_csv(masks / "threshold_evaluation.csv", index=False)
    # First maximum retains the recorded threshold-grid tie rule.
    best = table.loc[table.mean_dice.idxmax()]
    threshold = float(best["threshold"])
    thresholding.thresholding(oof, masks, threshold, 1, False)
    print(f"Selected threshold: {threshold:g} | mean Dice: {best.mean_dice:.6f}")
    return threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-a", type=int, required=True)
    parser.add_argument("--dataset-b", type=int, required=True,
                        help="A second prepared dataset with complementary validation splits")
    parser.add_argument("--images", type=Path, required=True, help="Two-channel images for both subsets")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--pretrained-model", type=Path, required=True,
                        help="Baseline model directory containing fold_0..fold_4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--train-iterations", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--thresholds", nargs="+", type=float, default=THRESHOLDS)
    args = parser.parse_args()
    if args.dataset_a == args.dataset_b:
        parser.error("Provide two distinct datasets")
    if any(not 0 <= value <= 1 for value in args.thresholds):
        parser.error("Thresholds must be between 0 and 1")
    if "nnUNet_preprocessed" not in os.environ:
        parser.error("Set the nnUNet_preprocessed environment variable")
    datasets = (args.dataset_a, args.dataset_b)
    cases = validation_cases(Path(os.environ["nnUNet_preprocessed"]), datasets)
    inputs = tuple(args.output / "inputs" / f"model_{i}" for i in (1, 2))
    outputs = tuple(args.output / "predictions" / f"model_{i}" for i in (1, 2))
    oof = args.output / "ensemble_oof" / f"{TRAINER}_{CONFIGURATION}"
    prepare_inputs(cases, args.images, args.labels, inputs)
    print(f"OOF validation: {len(cases[0])} + {len(cases[1])} cases")
    for dataset in datasets:
        run_finetuning(dataset, args.pretrained_model, args.epochs,
                       args.train_iterations, args.learning_rate)
    predict(datasets, cases, inputs, outputs, oof)
    optimize_threshold(oof, oof / "thresholding", args.labels, args.thresholds)


if __name__ == "__main__":
    main()
