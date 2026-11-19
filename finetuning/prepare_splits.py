"""Create complementary two-fold splits within the fine-tuning cohort."""

import argparse
import json
import random
from pathlib import Path


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
    """One prepared target dataset serves both CV halves and the all-case refit."""
    if output.exists():
        raise FileExistsError(output)
    all_cases = set(cases)
    if not half_a or not half_a < all_cases:
        raise ValueError("Calibration A must be a nonempty proper subset")
    half_b = all_cases-half_a
    splits = [dict(train=sorted(half_a), val=sorted(half_b)),
              dict(train=sorted(half_b), val=sorted(half_a))]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(splits, indent=2)+"\n")
    print(f"Two-fold CV: {len(half_a)}/{len(half_b)} complementary cases -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True,
                        help="labelsTr containing only fine-tuning cases")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path to the target dataset's new splits_final.json file")
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
