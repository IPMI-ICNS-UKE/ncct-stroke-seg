"""Select fine-tuning duration using pooled out-of-fold Dice on adaptation cases."""

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch
from inference.nnunet_predict import predict
from nnunet_extension.training import read_json, train_model


def select(dataset, source, output, epochs=(5, 10, 15, 20), iterations=20,
           main_lr=1e-3, norm_lr=5e-4, seed=20261002):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    epochs = sorted(set(epochs))
    if not epochs or epochs[0] < 1:
        raise ValueError('Durations must be positive')
    pp = Path(os.environ['nnUNet_preprocessed'])/dataset
    raw = Path(os.environ['nnUNet_raw'])/dataset
    splits = read_json(pp/'splits_final.json')
    cases = {p.name.removesuffix('.nii.gz') for p in (raw/'labelsTr').glob('*.nii.gz')}
    if len(splits) != 2 or set(splits[0]['train']) != set(splits[1]['val']) or set(splits[1]['train']) != set(splits[0]['val']):
        raise ValueError('Expected complementary two-fold adaptation splits')
    if set(splits[0]['train']) & set(splits[0]['val']) or set(splits[0]['train']) | set(splits[0]['val']) != cases:
        raise ValueError('CV must partition exactly the adaptation dataset')
    output.mkdir(parents=True)
    records = []
    for fold, split in enumerate(splits):
        trainer = train_model(dataset, fold, source=source, epochs=max(epochs), iterations=iterations,
            main_lr=main_lr, norm_lr=norm_lr, seed=seed, snapshot_epochs=epochs,
            output_root=output/'training')
        model_folder = Path(trainer.output_folder)
        del trainer
        gc.collect(); torch.cuda.empty_cache()
        for epoch in epochs:
            prediction_folder = output/'predictions'/f'epoch_{epoch:03d}'/f'fold_{fold}'
            predict(model_folder/f'checkpoint_epoch_{epoch:03d}.pth', raw/'imagesTr',
                    prediction_folder, cases=split['val'])
            for case in split['val']:
                pred = sitk.GetArrayFromImage(sitk.ReadImage(str(prediction_folder/f'{case}.nii.gz'))) > 0
                truth = sitk.GetArrayFromImage(sitk.ReadImage(str(raw/'labelsTr'/f'{case}.nii.gz'))) > 0
                if pred.shape != truth.shape:
                    raise ValueError(f'Array grid mismatch: {case}')
                dice = float(2*(pred & truth).sum()/(pred.sum()+truth.sum())) if truth.any() else 0.
                records.append(dict(epoch=epoch, fold=fold, case=case, dice=dice))
    summaries = []
    for epoch in epochs:
        selected = [r for r in records if r['epoch'] == epoch]
        assert len(selected) == len(cases) == len({r['case'] for r in selected})
        values = np.asarray([r['dice'] for r in selected])
        q1, median, q3 = np.quantile(values, [.25, .5, .75])
        summaries.append(dict(epochs=epoch, n=len(cases), mean_dice=float(values.mean()),
            median_dice=float(median), dice_q25=float(q1), dice_q75=float(q3)))
    best = max(summaries, key=lambda row: (row['median_dice'], -row['epochs']))
    for name, rows in [('case_dice.csv', records), ('duration_results.csv', summaries)]:
        with (output/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = dict(selected_epochs=best['epochs'], criterion='Highest pooled median Dice; shorter duration breaks ties',
        main_lr=main_lr, norm_lr=norm_lr, iterations_per_epoch=iterations, seed=seed, results=summaries)
    (output/'selection.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, nargs='+', default=[5, 10, 15, 20])
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--main-lr', type=float, default=1e-3)
    parser.add_argument('--norm-lr', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=20261002)
    args = parser.parse_args()
    select(args.dataset, args.source_checkpoint, args.output, args.epochs,
           args.iterations, args.main_lr, args.norm_lr, args.seed)


if __name__ == '__main__':
    main()
