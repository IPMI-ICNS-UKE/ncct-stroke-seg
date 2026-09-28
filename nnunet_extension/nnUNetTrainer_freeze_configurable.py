"""Configurable training duration and learning rate for the recorded frozen-encoder trainer."""

import os

import torch

from .nnUNetTrainer_freeze_test import nnUNetTrainer_freeze_test


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


class nnUNetTrainer_freeze_configurable(nnUNetTrainer_freeze_test):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = _positive_int("STROKE_FINETUNE_EPOCHS", 20)
        self.num_iterations_per_epoch = _positive_int("STROKE_FINETUNE_ITERATIONS", 60)
        self.initial_lr = float(os.environ.get("STROKE_FINETUNE_LR", "0.001"))
        if self.initial_lr <= 0:
            raise ValueError("STROKE_FINETUNE_LR must be positive")
