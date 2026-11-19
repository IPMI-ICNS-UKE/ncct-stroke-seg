"""Predict with a single full-cohort or adapted symmetry-comparison checkpoint."""

import argparse
from pathlib import Path
import os
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunet_extension.model import build_network


def load_model(checkpoint, device='cpu'):
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    args = saved['init_args']
    pm = PlansManager(args['plans'])
    cm = pm.get_configuration(args['configuration'])
    dataset = args['dataset_json']
    if len(dataset['channel_names']) != 4:
        raise ValueError('Expected a four-channel comparison checkpoint')
    network = build_network(pm, cm, 4, pm.get_label_manager(dataset).num_segmentation_heads, False)
    network.load_state_dict(saved['network_weights'], strict=True)
    return network.eval().to(device), saved, pm, dataset


def predict(checkpoint, input_folder, output_folder, device='cuda', save_probabilities=False, workers=2, cases=None):
    input_folder, output_folder = Path(input_folder), Path(output_folder)
    if not input_folder.is_dir():
        raise NotADirectoryError(input_folder)
    if output_folder.exists() and any(output_folder.iterdir()):
        raise FileExistsError(f'Use an empty prediction directory: {output_folder}')
    if workers < 1:
        raise ValueError('workers must be positive')
    torch.set_num_threads(1)
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use --device cpu for CPU prediction')
        torch.set_autocast_dtype('cuda', torch.bfloat16)
    network, saved, pm, dataset = load_model(checkpoint, device)
    predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_device=device.startswith('cuda'), device=torch.device(device), allow_tqdm=False)
    # Keep inference eager without changing compilation for subsequent training.
    previous_compile = os.environ.get('nnUNet_compile')
    os.environ['nnUNet_compile'] = 'false'
    try:
        predictor.manual_initialization(network, pm, pm.get_configuration(saved['init_args']['configuration']),
            [saved['network_weights']], dataset, 'nnUNetTrainerSymmetryComparison',
            saved.get('inference_allowed_mirroring_axes'))
    finally:
        if previous_compile is None:
            os.environ.pop('nnUNet_compile', None)
        else:
            os.environ['nnUNet_compile'] = previous_compile
    inputs = str(input_folder) if cases is None else [
        [str(input_folder/f'{case}_{i:04d}.nii.gz') for i in range(4)] for case in cases]
    if workers == 1:
        output_folder.mkdir(parents=True, exist_ok=True)
        predictor.predict_from_files_sequential(inputs, str(output_folder),
            save_probabilities=save_probabilities, overwrite=False)
    else:
        predictor.predict_from_files(inputs, str(output_folder), save_probabilities=save_probabilities,
            overwrite=False, num_processes_preprocessing=workers, num_processes_segmentation_export=workers)
    return output_folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--save-probabilities', action='store_true')
    args = parser.parse_args()
    predict(args.checkpoint, args.input, args.output, args.device, args.save_probabilities, args.workers)


if __name__ == '__main__':
    main()
