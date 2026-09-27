"""Install the unchanged custom trainer into the active nnU-Net package."""

from pathlib import Path
import shutil
import nnunetv2


def main() -> None:
    source = Path(__file__).with_name("nnUNetTrainer_freeze_test.py")
    destination = (
        Path(nnunetv2.__file__).parent
        / "training/nnUNetTrainer/variants/training_length"
        / source.name
    )
    if destination.exists():
        if destination.read_bytes() == source.read_bytes():
            print(f"Trainer already installed: {destination}")
            return
        raise FileExistsError(f"Different trainer already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"Installed trainer: {destination}")


if __name__ == "__main__":
    main()
