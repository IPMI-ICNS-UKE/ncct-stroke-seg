"""Preprocess one four-channel dataset using the study plan or a source model's plan."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import shutil
import numpy as np
import SimpleITK as sitk
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunet_extension.training import PLAN, read_json


def preprocess_case(job):
    case, raw, output, plans, dataset = job
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    pm = PlansManager(plans)
    preprocessor = DefaultPreprocessor(verbose=False)
    data, seg, properties = preprocessor.run_case(
        [str(raw/'imagesTr'/f'{case}_{i:04d}.nii.gz') for i in range(4)],
        str(raw/'labelsTr'/f'{case}.nii.gz'), pm, pm.get_configuration('3d_fullres'), dataset)
    if not np.isfinite(data).all() or not np.isin(seg, [-1, 0, 1]).all():
        raise ValueError(f'Invalid image or binary annotation: {case}')
    properties['class_locations'] = preprocessor._sample_foreground_locations(seg, [1])
    data, seg = data.astype(np.float32), seg.astype(np.int8)
    prefix = output/plans['configurations']['3d_fullres']['data_identifier']/case
    np.savez_compressed(str(prefix)+'.npz', data=data, seg=seg)
    np.save(str(prefix)+'.npy', data)
    np.save(str(prefix)+'_seg.npy', seg)
    with Path(str(prefix)+'.pkl').open('wb') as stream:
        pickle.dump(properties, stream, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.copy2(raw/'labelsTr'/f'{case}.nii.gz', output/'gt_segmentations'/f'{case}.nii.gz')
    return properties


def prepare(dataset_name, source_checkpoint=None, workers=4):
    raw = Path(os.environ['nnUNet_raw'])/dataset_name
    output = Path(os.environ['nnUNet_preprocessed'])/dataset_name
    if output.exists():
        raise FileExistsError(f'Use a new preprocessed dataset directory: {output}')
    dataset = read_json(raw/'dataset.json')
    if len(dataset['channel_names']) != 4 or dataset['labels'] != {'background': 0, 'stroke': 1}:
        raise ValueError('Expected four channels and binary background/stroke labels')
    cases = sorted(p.name.removesuffix('.nii.gz') for p in (raw/'labelsTr').glob('*.nii.gz'))
    if not cases or len(cases) != dataset['numTraining']:
        raise ValueError('Training labels and numTraining disagree')
    expected = {f'{c}_{i:04d}.nii.gz' for c in cases for i in range(4)}
    if {p.name for p in (raw/'imagesTr').glob('*.nii.gz')} != expected:
        raise ValueError('Training CT channels and labels do not match')
    if source_checkpoint is None:
        plans = read_json(Path(__file__).parents[1]/'nnunet_extension/plans.json')
    else:
        import torch
        saved = torch.load(source_checkpoint, map_location='cpu', mmap=True, weights_only=False)
        plans = copy.deepcopy(saved['init_args']['plans'])
        plans['comparison_hparams'] = plans.pop('frontier_hparams', plans.get('comparison_hparams', {}))
    plans.update(dataset_name=dataset_name, plans_name=PLAN)
    cm = plans['configurations']['3d_fullres']
    cm['data_identifier'] = PLAN+'_3d_fullres'
    if cm['normalization_schemes'] != ['CTNormalization', 'CTNormalization', 'NoNormalization', 'NoNormalization']:
        raise ValueError('The source plan is not a four-channel comparison plan')
    if plans['foreground_intensity_properties_per_channel']['0'] != plans['foreground_intensity_properties_per_channel']['1']:
        raise ValueError('CT and mirror must use the same normalization statistics')
    (output/cm['data_identifier']).mkdir(parents=True)
    (output/'gt_segmentations').mkdir()
    for name, value in ((f'{PLAN}.json', plans), ('dataset.json', dataset)):
        (output/name).write_text(json.dumps(value, indent=2)+'\n')
    jobs = [(case, raw, output, plans, dataset) for case in cases]
    if workers == 1:
        properties = list(map(preprocess_case, jobs))
    else:
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            properties = list(pool.map(preprocess_case, jobs))
    fingerprint = dict(spacings=[p['spacing'] for p in properties],
        shapes_after_crop=[p['shape_after_cropping_and_before_resampling'] for p in properties],
        foreground_intensity_properties_per_channel=plans['foreground_intensity_properties_per_channel'],
        median_relative_size_after_cropping=float(np.median([
            np.prod(p['shape_after_cropping_and_before_resampling'])/np.prod(p['shape_before_cropping']) for p in properties])))
    (output/'dataset_fingerprint.json').write_text(json.dumps(fingerprint, indent=2)+'\n')
    if (raw/'splits_final.json').exists():
        shutil.copy2(raw/'splits_final.json', output/'splits_final.json')
    print(f'Prepared {len(cases)} cases: {output}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--source-checkpoint', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    prepare(args.dataset, args.source_checkpoint, args.workers)


if __name__ == '__main__':
    main()
