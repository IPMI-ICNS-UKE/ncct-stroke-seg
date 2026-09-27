import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from inference import nnunet_predict
from threshold_selection import full_evaluation, thresholding

ROOT = Path(os.environ["NNUNET_WORKSPACE"])
PREPROCESSED, RESULTS = ROOT / "nnUNet_preprocessed", ROOT / "nnUNet_results"
HYPER = ROOT / "nnUNet_predict/Dataset_ISLES_finetuning_hyper"
IMAGES, LABELS = HYPER / "imagesVal_flipped", HYPER / "labelsVal_flipped"
DATASETS = ((36, "Dataset036_ISLES_flipped_finetuneValidation"),
            (37, "Dataset037_ISLES_flipped_finetuneValidation2"))
TRAINER, PLAN, CONFIG = "nnUNetTrainer_freeze_test", "nnUNetResEncUNetMPlans", "3d_fullres"
FOLDS = tuple(map(str, range(5)))
THRESHOLDS = (.5, .4, .3, .2, .1, .09, .08, .07, .06, .05, .04, .03, .02, .01)

WORK = HYPER / "predictions/threshold_search"
INPUTS = (HYPER / "imagesVal_flipped_model1_val", HYPER / "imagesVal_flipped_model2_val")
OUTPUTS = (WORK / "model_1", WORK / "model_2")
IDENTIFIER = f"{TRAINER}_{CONFIG}"
OOF = WORK / "ensemble_oof" / IDENTIFIER
MASKS = OOF / "thresholding"
PRETRAINED = RESULTS / "Dataset023_UKEAISDfull_flipped/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres"


def validation_cases():
    splits = []
    for _, name in DATASETS:
        with (PREPROCESSED / name / "splits_final.json").open() as file:
            folds = json.load(file)
        if len(folds) != 5 or any(fold != folds[0] for fold in folds):
            raise RuntimeError(f"Unexpected folds in {name}/splits_final.json")
        splits.append(folds[0])

    a, b = splits
    if set(a["train"]) != set(b["val"]) or set(a["val"]) != set(b["train"]):
        raise RuntimeError("Dataset036 and Dataset037 are not complementary")
    return tuple(map(sorted, (a["val"], b["val"])))


def link_exact(sources, destination, suffixes=(".nii.gz", ".npz")):
    """Create only missing links and refuse to remove or replace existing data."""
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


def prepare_inputs(case_groups):
    for cases, destination in zip(case_groups, INPUTS):
        sources = []
        for case in cases:
            modalities = sorted(IMAGES.glob(f"{case}_*.nii.gz"))
            if len(modalities) != 2 or not (LABELS / f"{case}.nii.gz").is_file():
                raise RuntimeError(f"Incomplete image or label data for {case}")
            sources += modalities
        link_exact(sources, destination, (".nii.gz",))


def train():
    for dataset, _ in DATASETS:
        for fold in FOLDS:
            weights = PRETRAINED / f"fold_{fold}/checkpoint_best.pth"
            subprocess.run([
                "nnUNetv2_train", str(dataset), CONFIG, fold, "-device", "cuda",
                "-p", PLAN, "-tr", TRAINER, "-pretrained_weights", str(weights)
            ], check=True)


def predict(case_groups):
    prediction_folders, oof_sources = [], []
    for (dataset, _), cases, input_folder, output in zip(DATASETS, case_groups, INPUTS, OUTPUTS):
        nnunet_predict.run_prediction(
            dataset, FOLDS, input_folder, output, LABELS, CONFIG, TRAINER, PLAN, "1"
        )
        folder = output / IDENTIFIER
        required = [folder / f"{case}{suffix}" for case in cases for suffix in (".nii.gz", ".npz")]
        if any(not path.is_file() for path in required):
            raise RuntimeError(f"Missing validation predictions in {folder}")
        found = {p.name for p in folder.iterdir() if p.name.endswith((".nii.gz", ".npz"))}
        if found != {p.name for p in required}:
            raise RuntimeError(f"Prediction set mismatch in {folder}")
        prediction_folders.append(folder)
        oof_sources += required
    link_exact(oof_sources, OOF)
    return prediction_folders


def optimize_threshold():
    rows, best = [], None
    for value in THRESHOLDS:
        thresholding.thresholding(OOF, MASKS, LABELS, value, 1, False)
        metrics = full_evaluation.full_evaluation(MASKS, LABELS)
        if metrics is None:
            raise RuntimeError(f"No cases evaluated at threshold {value}")
        mean_dice, median_dice, hd_accuracy, mean_fp, mean_ap = metrics
        rows.append(dict(threshold=value, mean_dice=mean_dice, median_dice=median_dice,
                         hd_accuracy=hd_accuracy, mean_fp=mean_fp, mAP=mean_ap))
        if best is None or mean_dice > best[1]:
            best = value, mean_dice
    pd.DataFrame(rows).to_csv(MASKS / "threshold_evaluation.csv", index=False)
    print(f"Best threshold: {best[0]} | Best mean Dice: {best[1]:.6f}")
    return best


def save_run(best_threshold):
    thresholding.thresholding(OOF, MASKS, LABELS, best_threshold, 1, False)
    destination = HYPER / "expected_run"
    for number, (_, name) in enumerate(DATASETS, 1):
        model = RESULTS / name / f"{TRAINER}__{PLAN}__{CONFIG}"
        shutil.copytree(model, destination / f"model_{number}", dirs_exist_ok=True)
    shutil.copytree(OOF, destination / "predictions", dirs_exist_ok=True)


def main(iterations):
    cases = validation_cases()
    prepare_inputs(cases)
    print(f"Prepared complementary OOF inputs: {len(cases[0])} + {len(cases[1])} cases")
    for iteration in range(iterations):
        print(f"Threshold-search iteration {iteration}")
        train()
        predict(cases)
        best_threshold, _ = optimize_threshold()  # Selection target: mean Dice.
        if .02 <= best_threshold < .04:
            save_run(best_threshold)
            print(f"Expected threshold found; run saved to {HYPER / 'expected_run'}")
            return
    print("No run in the expected threshold range was found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    main(parser.parse_args().iterations)
