"""Shared training entry point; no nnU-Net installation files are modified."""

from pathlib import Path
import copy
import json
import os
import random
import numpy as np
import torch

PLAN = 'SymmetryResEncMPlans'


def read_json(path):
    return json.loads(Path(path).read_text())


def train_model(dataset, fold='all', foundation=None, source=None, epochs=100,
                iterations=250, main_lr=1e-3, norm_lr=5e-4, seed=20261002,
                snapshot_epochs=(), output_root=None, resume=False):
    if epochs < 1 or iterations < 1 or not np.isfinite(main_lr) or main_lr <= 0 or not 0 < norm_lr < float('inf'):
        raise ValueError('Epochs, iterations and learning rates must be positive and finite')
    if output_root is not None:
        os.environ['nnUNet_results'] = str(Path(output_root).resolve())
    from .trainer import nnUNetTrainerSymmetryComparison, nnUNetTrainerSymmetryFineTuning
    folder = Path(os.environ['nnUNet_preprocessed'])/dataset
    plans = read_json(folder/f'{PLAN}.json')
    dataset_json = read_json(folder/'dataset.json')
    cfg = plans['comparison_hparams']
    if source is None:
        if foundation is None or not Path(foundation).is_file():
            raise FileNotFoundError('Supply the downloaded nnFoundationCNN checkpoint')
        plans['pretrain_info']['checkpoint_path'] = str(Path(foundation).resolve())
        cfg.update(epochs=epochs, updates_per_epoch=iterations, new_lr=main_lr)
        if epochs <= cfg['warmup_epochs']:
            raise ValueError('Source training must extend beyond warmup')
        trainer_type = nnUNetTrainerSymmetryComparison
        saved = None
    else:
        saved = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
        original = saved['init_args']['plans']
        old = original['configurations'][saved['init_args']['configuration']]
        current = plans['configurations']['3d_fullres']
        # Keep target preprocessing on the source model's scale and geometry.
        for key in ('architecture', 'spacing', 'patch_size', 'normalization_schemes', 'use_mask_for_norm'):
            if current[key] != old[key]:
                raise ValueError(f'Target/source plans differ in {key}; prepare with --source-checkpoint')
        if plans['foreground_intensity_properties_per_channel'] != original['foreground_intensity_properties_per_channel']:
            raise ValueError('Target CT normalization differs from source checkpoint')
        plans['finetuning'] = dict(epochs=epochs, iterations=iterations,
            main_lr=main_lr, norm_lr=norm_lr, snapshot_epochs=list(snapshot_epochs))
        trainer_type = nnUNetTrainerSymmetryFineTuning
    plans['continue_training'] = resume
    fold = 'all' if str(fold) == 'all' else int(fold)
    if fold != 'all' and fold not in range(5):
        raise ValueError('Use fold 0–4 or all')
    expected = (Path(os.environ['nnUNet_results'])/dataset/
                f'{trainer_type.__name__}__{PLAN}__3d_fullres'/f'fold_{fold}')
    if (expected/'checkpoint_final.pth').exists():
        raise FileExistsError(f'Completed model exists: {expected}')
    latest = expected/'checkpoint_latest.pth'
    if latest.exists() and not resume:
        raise FileExistsError(f'Unfinished model exists; use --resume: {expected}')
    if resume and not latest.is_file():
        raise FileNotFoundError(f'No checkpoint to resume: {latest}')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    trainer = trainer_type(copy.deepcopy(plans), '3d_fullres', fold, dataset_json)
    if resume:
        trainer.load_checkpoint(str(latest))
    elif saved is not None:
        trainer.initialize()
        trainer.raw_network().load_state_dict(saved['network_weights'], strict=True)
    trainer.run_training()
    print(f'Final checkpoint: {expected / "checkpoint_final.pth"}')
    return trainer
