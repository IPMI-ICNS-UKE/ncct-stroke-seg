"""Create four-channel nnU-Net training or inference data from prepared CT pairs."""

import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import SimpleITK as sitk


def prepare(source, target, dataset_name, training=False, label_filename='mask.nii.gz'):
    source, target = Path(source).resolve(), Path(target).resolve()
    if target == source or source in target.parents or target in source.parents:
        raise ValueError('Source and target must be separate directories')
    if Path(dataset_name).name != dataset_name:
        raise ValueError('Dataset name must be a folder name')
    destination = target/dataset_name
    if destination.exists():
        raise FileExistsError(destination)
    cases = sorted(p for p in source.iterdir() if p.is_dir() and (p/'image.nii.gz').is_file())
    if not cases:
        raise ValueError('No prepared case folders found')
    # Check required files first. Folder names are the case IDs.
    for case in cases:
        if not (case/'image_flipped.nii.gz').is_file() or (training and not (case/label_filename).is_file()):
            raise FileNotFoundError(f'Missing CT pair or training label: {case}')
    images = destination/('imagesTr' if training else 'imagesVal')
    labels = destination/('labelsTr' if training else 'labelsVal')
    images.mkdir(parents=True)
    labels.mkdir()
    for case in cases:
        native = sitk.ReadImage(str(case/'image.nii.gz'))
        mirrored = sitk.ReadImage(str(case/'image_flipped.nii.gz'))
        if (native.GetSize(), native.GetSpacing(), native.GetDirection(), native.GetOrigin()) != (
                mirrored.GetSize(), mirrored.GetSpacing(), mirrored.GetDirection(), mirrored.GetOrigin()):
            raise ValueError(f'CT pair geometry differs: {case}')
        if not np.isclose(abs(native.GetDirection()[0]), 1):
            raise ValueError(f'Stored x axis must be left-right: {case}')
        data = [sitk.GetArrayFromImage(image) for image in (native, mirrored)]
        if not all(np.isfinite(a).all() and a.min() >= 0 and a.max() <= 80 for a in data):
            raise ValueError(f'Expected finite, skull-stripped CT clipped to 0-80 HU: {case}')
        if not np.allclose(data[0][:, :, ::-1], data[1], atol=1e-4, rtol=0):
            raise ValueError(f'Mirrored CT does not match the atlas-space reflection: {case}')
        for channel, name in enumerate(('image.nii.gz', 'image_flipped.nii.gz')):
            shutil.copy2(case/name, images/f'{case.name}_{channel:04d}.nii.gz')
            # Support comes from the clipped image, never from the lesion label.
            mask = sitk.GetImageFromArray((data[channel] > 0).astype(np.uint8))
            mask.CopyInformation(native)
            sitk.WriteImage(mask, str(images/f'{case.name}_{channel+2:04d}.nii.gz'), True)
        if (case/label_filename).is_file():
            shutil.copy2(case/label_filename, labels/f'{case.name}.nii.gz')
    dataset = dict(channel_names={'0': 'CT', '1': 'CT', '2': 'native_support', '3': 'mirror_support'},
        labels={'background': 0, 'stroke': 1}, file_ending='.nii.gz', name=dataset_name)
    if training:
        dataset['numTraining'] = len(cases)
    (destination/'dataset.json').write_text(json.dumps(dataset, indent=2)+'\n')
    print(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--label-filename', default='mask.nii.gz')
    parser.add_argument('--training', action='store_true')
    args = parser.parse_args()
    prepare(args.source, args.target, args.dataset_name, args.training, args.label_filename)


if __name__ == '__main__':
    main()
