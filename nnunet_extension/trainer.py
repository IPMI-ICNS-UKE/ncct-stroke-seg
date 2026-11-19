"""nnU-Net trainers for source training and selective domain fine-tuning."""
from pathlib import Path
import torch
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from .augmentation import PairedIntensityTransform
from .optimizers import configure_optimizer, ConstantLR
from .model import build_network, load_foundation


def settings(plans):
    # The second key exists only for loading unchanged historical checkpoints.
    return plans.get("comparison_hparams", plans.get("frontier_hparams", {}))


class WarmupGroupPoly:
    def __init__(self, optimizer, epochs, warmup):
        self.optimizer, self.epochs, self.warmup = optimizer, epochs, warmup
        self.peaks = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch):
        factor = ((epoch+1)/self.warmup if epoch < self.warmup else
                  ((self.epochs-epoch)/(self.epochs-self.warmup))**.9)
        for group, peak in zip(self.optimizer.param_groups, self.peaks):
            group["lr"] = peak*factor


class nnUNetTrainerSymmetryComparison(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        cfg = settings(plans)
        self.num_epochs = cfg.get("epochs", 100)
        self.num_iterations_per_epoch = cfg.get("updates_per_epoch", 250)
        self.num_val_iterations_per_epoch = cfg.get("validation_updates", 50)
        self.initial_lr = cfg.get("new_lr", 1e-3)
        self.weight_decay = cfg.get("weight_decay", 3e-5)
        self.save_every = 10
        self.snapshot_epochs = ()
        if device.type == "cuda" and cfg.get("amp_dtype", "bfloat16") == "bfloat16":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("This configuration requires a GPU supporting bfloat16")
            torch.set_autocast_dtype("cuda", torch.bfloat16)
            self.grad_scaler = None

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        return build_network(plans_manager, configuration_manager, num_input_channels,
                             num_output_channels, enable_deep_supervision)

    def raw_network(self):
        return getattr(self.network, "_orig_mod", self.network)

    def configure_optimizers(self):
        cfg = settings(self.plans_manager.plans)
        load_foundation(self.network, self.plans_manager, self.configuration_manager)
        encoder = list(self.network.encoder.parameters())
        encoder_ids = {id(p) for p in encoder}
        new = [p for p in self.network.parameters() if id(p) not in encoder_ids]
        optimizer = torch.optim.AdamW([
            dict(params=new, lr=cfg.get("new_lr", 1e-3), name="decoder_and_comparison"),
            dict(params=encoder, lr=cfg.get("encoder_lr", 1e-4), name="shared_encoder")],
            weight_decay=self.weight_decay, betas=(.9, .98), fused=self.device.type == "cuda")
        return optimizer, WarmupGroupPoly(optimizer, self.num_epochs, cfg.get("warmup_epochs", 38))

    @staticmethod
    def get_training_transforms(*args, **kwargs):
        base = nnUNetTrainer.get_training_transforms(*args, **kwargs)
        transforms, inserted = [], False
        intensity_names = {"GaussianNoiseTransform", "GaussianBlurTransform", "MultiplicativeBrightnessTransform",
                           "ContrastTransform", "SimulateLowResolutionTransform", "GammaTransform"}
        for transform in base.transforms:
            if isinstance(transform, RandomTransform) and type(transform.transform).__name__ in intensity_names:
                if not inserted:
                    transforms.append(PairedIntensityTransform())
                    inserted = True
            else:
                transforms.append(transform)
        assert inserted
        return ComposeTransforms(transforms)

    def plot_network_architecture(self):
        self.print_to_log_file("Shared ResEnc encoder, local comparison at stages 1/2, one decoder.")


    def save_snapshot(self, filename, epoch):
        torch.save(dict(network_weights=self.raw_network().state_dict(), current_epoch=epoch,
            init_args=self.my_init_kwargs, trainer_name=type(self).__name__,
            inference_allowed_mirroring_axes=self.inference_allowed_mirroring_axes), filename)

    def save_checkpoint(self, filename):
        name = Path(filename).name
        if name == 'checkpoint_best.pth':
            return
        if name == 'checkpoint_final.pth':
            self.save_snapshot(filename, self.current_epoch+1)
        else:
            super().save_checkpoint(filename)

    def on_epoch_end(self):
        epoch = self.current_epoch+1
        if epoch in self.snapshot_epochs:
            self.save_snapshot(Path(self.output_folder)/f'checkpoint_epoch_{epoch:03d}.pth', epoch)
        super().on_epoch_end()


class nnUNetTrainerSymmetryFineTuning(nnUNetTrainerSymmetryComparison):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        cfg = plans['finetuning']
        self.num_epochs = cfg['epochs']
        self.num_iterations_per_epoch = cfg['iterations']
        self.num_val_iterations_per_epoch = 10
        self.snapshot_epochs = tuple(cfg.get('snapshot_epochs', []))

    def configure_optimizers(self):
        cfg = self.plans_manager.plans['finetuning']
        optimizer = configure_optimizer(self.network, cfg['main_lr'], cfg['norm_lr'], self.weight_decay)
        self.print_to_log_file('Fine-tuning groups:',
            [(g['name'], sum(p.numel() for p in g['params']), g['lr']) for g in optimizer.param_groups])
        return optimizer, ConstantLR()
