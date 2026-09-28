"""nnU-Net trainer with a frozen encoder and configurable fine-tuning settings."""

import os

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


class nnUNetTrainer_freeze_configurable(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = _positive_int("STROKE_FINETUNE_EPOCHS", 20)
        self.num_iterations_per_epoch = _positive_int("STROKE_FINETUNE_ITERATIONS", 60)
        self.initial_lr = float(os.environ.get("STROKE_FINETUNE_LR", "0.001"))
        if self.initial_lr <= 0:
            raise ValueError("STROKE_FINETUNE_LR must be positive")

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        network = nnUNetTrainer.build_network_architecture(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, num_output_channels, enable_deep_supervision
        )
        # The base trainer moves the network to its device before configuring the optimizer.
        print("Freezing Encoder for Finetuning...")
        for param in network.encoder.parameters():
            param.requires_grad = False
        return network
