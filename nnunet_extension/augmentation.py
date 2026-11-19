"""Photometric changes share parameters and statistics across the CT pair."""
import torch
from torch.nn import functional as F
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform


class PairedIntensityTransform(ImageOnlyTransform):
    def _apply_to_image(self, img, **params):
        ct = img[:2].clone()
        support = (img[2:4] > 0.5).to(ct.dtype)
        # Shared blur, with no mixing of through-plane slices or support masks.
        if torch.rand(()) < 0.2:
            sigma = float(torch.empty(()).uniform_(0.5, 1.))
            a = torch.arange(-2, 3, dtype=ct.dtype, device=ct.device)
            k = torch.exp(-0.5*(a/sigma)**2)
            k = k/k.sum()
            kernel = (k[:, None]*k[None, :])[None, None, None].repeat(2, 1, 1, 1, 1)
            ct = F.conv3d(F.pad(ct[None], (2, 2, 2, 2, 0, 0), mode="replicate"), kernel, groups=2)[0]
        if torch.rand(()) < 0.15:
            ct *= float(torch.empty(()).uniform_(0.85, 1.15))
        if torch.rand(()) < 0.15:
            center = (ct*support).sum()/support.sum().clamp_min(1)
            factor = float(torch.empty(()).uniform_(0.85, 1.15))
            ct = (ct-center)*factor+center
        if torch.rand(()) < 0.25:
            scale = float(torch.empty(()).uniform_(0.7, 1.))
            shape = ct.shape[1:]
            small = [shape[0], max(1, round(shape[1]*scale)), max(1, round(shape[2]*scale))]
            ct = F.interpolate(F.interpolate(ct[None], size=small, mode="nearest"),
                               size=shape, mode="trilinear", align_corners=False)[0]
        if torch.rand(()) < 0.3:
            gamma = float(torch.empty(()).uniform_(0.85, 1.2))
            lo, hi = ct.min(), ct.max()
            mean, std = ct.mean(), ct.std()
            ct = ((ct-lo)/(hi-lo).clamp_min(1e-6)).clamp(0, 1).pow(gamma)*(hi-lo)+lo
            ct = (ct-ct.mean())/ct.std().clamp_min(1e-6)*std+mean
        # Shared noise field avoids manufacturing independent hemispheric noise.
        # Native scans already contain spatially different acquisition noise.
        if torch.rand(()) < 0.1:
            std = float(torch.empty(()).uniform_(0., 0.1))
            ct += torch.randn_like(ct[:1])*std
        return torch.cat([ct, support], dim=0)
