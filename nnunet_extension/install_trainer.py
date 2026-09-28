"""Install the configurable trainer into the active nnU-Net environment."""

from pathlib import Path
import shutil
import nnunetv2


def main() -> None:
    destination_dir = (Path(nnunetv2.__file__).parent
                       / "training/nnUNetTrainer/variants/training_length")
    source = Path(__file__).with_name("nnUNetTrainer_freeze_configurable.py")
    destination = destination_dir / source.name
    if destination.exists() and destination.read_bytes() != source.read_bytes():
        raise FileExistsError(f"Different trainer already exists: {destination}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
        print(f"Installed trainer: {destination}")
    else:
        print(f"Trainer already installed: {destination}")


if __name__ == "__main__":
    main()
