"""Create complementary calibration splits and a final all-fine-tuning split."""

import argparse
import json
import random
from pathlib import Path

from finetuning.train import FOLDS


def case_ids(labels: Path) -> list[str]:
    """Read exact nnU-Net case IDs from a fine-tuning-only labelsTr folder."""
    if not labels.is_dir():
        raise NotADirectoryError(labels)
    cases = sorted(path.name.removesuffix(".nii.gz") for path in labels.glob("*.nii.gz")
                   if path.is_file())
    if len(cases) < 2:
        raise ValueError("Expected at least two fine-tuning labels")
    return cases


def calibration_half(cases: list[str], seed: int | None, case_file: Path | None) -> set[str]:
    if case_file is None:
        if seed is None:
            raise ValueError("Provide a seed or a calibration-A case list")
        return set(random.Random(seed).sample(cases, (len(cases) + 1) // 2))
    chosen = [line.strip() for line in case_file.read_text().splitlines() if line.strip()]
    if len(chosen) != len(set(chosen)) or not 0 < len(chosen) < len(cases):
        raise ValueError("Calibration-A case list must be a nonempty, unique proper subset")
    if unknown := set(chosen) - set(cases):
        raise ValueError(f"Calibration-A cases not found in labels: {sorted(unknown)}")
    return set(chosen)


def write_splits(output: Path, cases: list[str], half_a: set[str]) -> None:
    """Write five identical entries per dataset, as expected by the trainer."""
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    all_cases = set(cases)
    if not half_a or not half_a < all_cases:
        raise ValueError("Calibration A must be a nonempty proper subset of fine-tuning cases")
    half_b = all_cases - half_a
    for name, train, val in (
        ("calibration_a", half_a, half_b),
        ("calibration_b", half_b, half_a),
        ("final", all_cases, all_cases),
    ):
        destination = output / name
        destination.mkdir(parents=True)
        folds = [{"train": sorted(train), "val": sorted(val)} for _ in FOLDS]
        (destination / "splits_final.json").write_text(json.dumps(folds, indent=2) + "\n")
        print(f"{name}: {len(train)} train, {len(val)} val -> {destination / 'splits_final.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True,
                        help="labelsTr containing only fine-tuning cases")
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for three splits_final.json files")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--seed", type=int, help="Seed for an approximately equal random split")
    choice.add_argument("--calibration-a-cases", type=Path,
                        help="Text file with exact nnU-Net case IDs, one per line")
    args = parser.parse_args()
    cases = case_ids(args.labels)
    half_a = calibration_half(cases, args.seed, args.calibration_a_cases)
    write_splits(args.output, cases, half_a)


if __name__ == "__main__":
    main()
