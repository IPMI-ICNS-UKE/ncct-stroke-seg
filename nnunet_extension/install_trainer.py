"""Install the recorded trainer and its configurable subclass."""

from pathlib import Path
import shutil
import nnunetv2


def main() -> None:
    destination_dir = (Path(nnunetv2.__file__).parent
                       / "training/nnUNetTrainer/variants/training_length")
    sources = [Path(__file__).with_name(name) for name in
               ("nnUNetTrainer_freeze_test.py", "nnUNetTrainer_freeze_configurable.py")]
    # Check both before writing either; never replace an existing trainer.
    for source in sources:
        destination = destination_dir / source.name
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise FileExistsError(f"Different trainer already exists: {destination}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        destination = destination_dir / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
            print(f"Installed trainer: {destination}")
        else:
            print(f"Trainer already installed: {destination}")


if __name__ == "__main__":
    main()
