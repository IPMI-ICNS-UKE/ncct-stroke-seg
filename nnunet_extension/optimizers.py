"""Selected Boston/ISLES optimizer: constant LR and restricted adaptation.

These functions contain the parameter selection used by the retained models.
They do not launch training or choose a data split. Boston: 10 epochs; ISLES:
40 epochs; both use 20 updates/epoch and batch size 2 (saved configurations).
"""
import torch
from torch import nn

MAIN_LR = 1e-3
EARLY_NORM_LR = 5e-4
WEIGHT_DECAY = 3e-5


def configure_optimizer(network, main_lr=MAIN_LR, norm_lr=EARLY_NORM_LR,
                        weight_decay=WEIGHT_DECAY):
    network = getattr(network, "_orig_mod", network)
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    assigned, groups = set(), []

    def add(name, parameters, lr):
        selected = []
        for parameter in parameters:
            if id(parameter) not in assigned:
                assigned.add(id(parameter))
                parameter.requires_grad_(True)
                selected.append(parameter)
        if selected:
            groups.append(dict(name=name, params=selected, lr=lr))

    add("comparison", network.adapters.parameters(), main_lr)
    # decoder.encoder references the shared encoder too. Select only these
    # blocks, so the encoder convolution weights remain exactly frozen.
    last = (network.decoder.stages[-1], network.decoder.seg_layers[-1])
    add("decoder", (p for block in last for p in block.parameters()), main_lr)
    early = (network.encoder.stem, *list(network.encoder.stages)[:3])
    add("early_encoder_norm", (p for block in early for module in block.modules()
        if isinstance(module, nn.InstanceNorm3d) for p in module.parameters()), norm_lr)
    assert sum(p.numel() for group in groups for p in group["params"]) == 53286
    assert all(not p.requires_grad for p in network.encoder.parameters() if p.ndim >= 3)
    return torch.optim.AdamW(groups, betas=(.9, .98), weight_decay=weight_decay,
                             fused=next(network.parameters()).device.type == "cuda")


class ConstantLR:
    def step(self, epoch):
        pass
