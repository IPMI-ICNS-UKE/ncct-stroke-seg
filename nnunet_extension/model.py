"""Shared-encoder nnU-Net with local contralateral feature comparison.

Four inputs: CT, mirrored CT, and their image-derived support masks. The
CT pair enters one shared encoder; masks control valid local comparisons.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def shifted(x, dy, dx):
    """output[y,x] = input[y+dy,x+dx]; pad, never wrap at an image boundary."""
    h, w = x.shape[-2:]
    padded = F.pad(x, (1, 1, 1, 1, 0, 0))
    return padded[..., 1+dy:1+dy+h, 1+dx:1+dx+w]


# A comparison module acts on downsampled encoder feature maps. It does not
# segment a hemisphere separately: it adds contralateral evidence to a skip map.
class LocalComparison(nn.Module):
    def __init__(self, channels, projection=16, feature_mm=2., sigma_mm=2.):
        super().__init__()
        self.embedding = nn.Conv3d(channels, projection, 1, bias=False)
        self.offsets = [(y, x) for y in [-1, 0, 1] for x in [-1, 0, 1]]
        self.priors = [-((y*feature_mm)**2+(x*feature_mm)**2)/(2*sigma_mm**2) for y, x in self.offsets]
        # Native, matched-reference, signed feature difference, cosine disparity,
        # fine and smoothed signed intensity differences, and valid support.
        self.fusion = nn.Sequential(nn.Conv3d(projection*3+4, 32, 1), nn.GroupNorm(4, 32),
                                    nn.GELU(), nn.Conv3d(32, 32, (1, 3, 3), padding=(0, 1, 1)), nn.GELU())
        self.correction = nn.Conv3d(32, channels, 1, bias=False)
        self.gate = nn.Conv3d(32, 1, 1)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.correction.weight)

    def compare(self, native, reference, intensity, support):
        # The same 1x1 projection places native/mirrored features in one space.
        n = self.embedding(native)
        r = self.embedding(reference)
        q = F.normalize(n.float(), dim=1, eps=1e-6)
        k = F.normalize(r.float(), dim=1, eps=1e-6)
        shape = native.shape[2:]
        # Average is a physically meaningful local intensity descriptor; masks
        # are averaged separately to exclude partly padded neighborhoods.
        # Intensities are CT-normalized; support channels remain unnormalized.
        hu = F.adaptive_avg_pool3d(intensity.float(), shape)
        masks = F.adaptive_avg_pool3d(support.float(), shape)
        native_valid = masks[:, 0:1] > 0.5
        ref_valid = (masks[:, 1:2] > 0.5).float()
        scores, valid = [], []
        # Search nine in-plane neighbors in the mirrored feature map. Matching
        # is soft (weighted); there is no z search or independent second encoder.
        for (dy, dx), prior in zip(self.offsets, self.priors):
            v = shifted(ref_valid, dy, dx) > 0.5
            score = 4.0*(q*shifted(k, dy, dx)).sum(1, keepdim=True) + prior
            scores.append(score.masked_fill(~v, -10000.))
            valid.append(v)
        valid = torch.cat(valid, 1)
        weights = torch.softmax(torch.cat(scores, 1), dim=1)*valid.float()
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        # Zero weights for all-invalid references: finite and exact fallback.
        matched = torch.zeros_like(r)
        matched_hu = torch.zeros_like(hu[:, 1:2])
        for idx, (dy, dx) in enumerate(self.offsets):
            w = weights[:, idx:idx+1]
            matched = matched + w.to(r.dtype)*shifted(r, dy, dx)
            matched_hu = matched_hu + w*shifted(hu[:, 1:2], dy, dx)
        available = native_valid.float()*valid.any(1, keepdim=True).float()
        signed_hu = (matched_hu-hu[:, 0:1])*available
        smooth_hu = F.avg_pool3d(F.pad(signed_hu, (1, 1, 1, 1, 0, 0), mode="replicate"), (1, 3, 3), stride=1)
        cosine = (1-(q*F.normalize(matched.float(), dim=1, eps=1e-6)).sum(1, keepdim=True))*available
        descriptor = torch.cat([n, matched, matched-n, cosine.to(n.dtype),
                                signed_hu.to(n.dtype), smooth_hu.to(n.dtype), available.to(n.dtype)], 1)
        hidden = self.fusion(descriptor)
        gate = torch.sigmoid(self.gate(hidden))*available.to(hidden.dtype)
        # The gate weights evidence reliability. Zero-initialized corrections
        # initially preserve the native features; training learns their changes.
        correction = self.alpha*gate*self.correction(hidden)
        return correction, gate, signed_hu, weights

    def forward(self, native, reference, intensity, support, keep):
        correction, _, _, _ = self.compare(native, reference, intensity, support)
        return native + keep*correction


class SymmetryComparisonNetwork(nn.Module):
    def __init__(self, base, projection=16, dropout=0.1, activation_checkpointing=False):
        super().__init__()
        self.encoder = base.encoder
        self.decoder = base.decoder
        self.adapters = nn.ModuleDict({"1": LocalComparison(self.encoder.output_channels[1], projection, 2., 2.),
                                       "2": LocalComparison(self.encoder.output_channels[2], projection, 4., 4.)})
        self.dropout = dropout
        self.activation_checkpointing = activation_checkpointing
        self.key_to_encoder = base.key_to_encoder
        self.key_to_stem = base.key_to_stem

    def _run(self, module, *args):
        if self.training and self.activation_checkpointing and torch.is_grad_enabled():
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(self, x):
        # CT channels remain the only inputs to the pretrained image encoder.
        assert x.shape[1] == 4, "Expected native CT, mirrored CT, and their two support masks"
        native = self._run(self.encoder.stem, x[:, 0:1])
        reference = self._run(self.encoder.stem, x[:, 1:2])
        keep = torch.ones((x.shape[0], 1, 1, 1, 1), device=x.device, dtype=native.dtype)
        if self.training:
            keep = (torch.rand((x.shape[0], 1, 1, 1, 1), device=x.device) >= self.dropout).to(native.dtype)
        # Reference features are needed only through encoder stage 2. Corrected
        # maps go to the decoder; native features continue down the encoder.
        skips = []
        for i, stage in enumerate(self.encoder.stages):
            native = self._run(stage, native)
            if i <= 2:
                reference = self._run(stage, reference)
            if str(i) in self.adapters:
                skip = self._run(self.adapters[str(i)], native, reference, x[:, :2], x[:, 2:4], keep)
            else:
                skip = native
            skips.append(skip)
        return self.decoder(skips)


def build_network(plans_manager, configuration_manager, num_input_channels, num_output_channels, deep_supervision=True):
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    if num_input_channels != 4:
        raise ValueError("Expected CT, mirrored CT, and two support channels")
    base = get_network_from_plans(configuration_manager.network_arch_class_name,
                                 configuration_manager.network_arch_init_kwargs,
                                 configuration_manager.network_arch_init_kwargs_req_import,
                                 1, num_output_channels, deep_supervision=deep_supervision)
    cfg = plans_manager.plans.get("comparison_hparams", plans_manager.plans.get("frontier_hparams", {}))
    return SymmetryComparisonNetwork(base, cfg.get("projection", 16), cfg.get("comparison_dropout", 0.1),
                        cfg.get("activation_checkpointing", False))


def load_foundation(net, plans_manager, configuration_manager):
    from nnunetv2.training.nnUNetTrainer.pretraining.dynamicPretrainedTrainer import DynamicPretrainedTrainer
    from nnunetv2.utilities.load_weights_utils import adapt_conv_kernel_size
    info = plans_manager.plans["pretrain_info"]
    arch = configuration_manager.network_arch_init_kwargs
    net, mismatch = DynamicPretrainedTrainer.load_pretrained_weights_dynamic(
        net, pretrained_weights_path=info["checkpoint_path"], pt_input_channels=1, downstream_input_channels=1,
        pt_input_patchsize=info["pt_used_patchsize"], downstream_input_patchsize=configuration_manager.patch_size,
        pt_key_to_encoder=info["key_to_encoder"], pt_key_to_stem=info["key_to_stem"],
        pt_keys_to_in_proj=tuple(info["keys_to_in_proj"]), pt_key_to_lpe=info["key_to_lpe"],
        target_kernel_sizes=arch["kernel_sizes"], target_n_stages=arch["n_stages"])
    source = torch.load(info["checkpoint_path"], weights_only=True, mmap=True, map_location="cpu")["network_weights"]
    matched, adapted = 0, 0
    for name, param in net.named_parameters():
        if not name.startswith("encoder."):
            continue
        expected = source[name]
        if expected.shape != param.shape:
            expected = adapt_conv_kernel_size(expected, list(param.shape[2:]), list(expected.shape[2:]))
            adapted += 1
        assert torch.equal(param.detach().cpu(), expected), f"Failed foundation transfer: {name}"
        matched += 1
    assert not mismatch and matched == 224
    return {"matched_encoder_tensors": matched, "adapted_kernels": adapted}
