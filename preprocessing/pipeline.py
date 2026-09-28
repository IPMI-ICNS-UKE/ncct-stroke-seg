"""Run the recorded CT and mask preprocessing stages in order."""

import argparse
from pathlib import Path


def run(source: Path, work: Path, output: Path, atlas: Path, device: str = "gpu") -> None:
    source, work, output, atlas = (path.resolve() for path in (source, work, output, atlas))
    if not source.is_dir() or not atlas.is_file():
        raise FileNotFoundError("Source directory and atlas file must exist")
    if any(a == b or a in b.parents or b in a.parents
           for a, b in ((source, work), (source, output), (work, output))):
        raise ValueError("Source, work, and output directories must be separate")
    if any(path.exists() and any(path.iterdir()) for path in (work, output)):
        raise FileExistsError("Work and output directories must be empty")
    cases = [path for path in source.iterdir() if path.is_dir()]
    if not cases or any(not (case / name).is_file()
                        for case in cases for name in ("image.nii.gz", "mask.nii.gz")):
        raise ValueError("Every source case needs image.nii.gz and mask.nii.gz")

    from .atlas_registration import atlas_registration
    from .create_flipped_channel import create_difference_map
    from .hu_clipping import hu_clipping
    from .skull_stripping import skull_stripping

    registered = work / "registered"
    stripped = work / "skull_stripped"
    atlas_registration(source, registered, atlas, convert_masks=True)
    skull_stripping(registered, stripped, convert_masks=True, device=device)
    hu_clipping(stripped, stripped, range_lower=0, range_upper=80)
    create_difference_map(stripped, output)
    print(f"Preprocessed {len(cases)} cases to {output}; transforms remain in {registered}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Case folders with image.nii.gz and mask.nii.gz")
    parser.add_argument("--work", type=Path, required=True, help="New directory for intermediate images and transforms")
    parser.add_argument("--output", type=Path, required=True, help="New directory for paired CT, mask, and mirrored CT")
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu", "mps"), default="gpu",
                        help="TotalSegmentator device (default: gpu)")
    args = parser.parse_args()
    run(args.source, args.work, args.output, args.atlas, device=args.device)


if __name__ == "__main__":
    main()
