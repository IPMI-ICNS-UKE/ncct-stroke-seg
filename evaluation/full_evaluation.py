"""Compute the recorded segmentation metrics against reference masks."""

import argparse
import csv
import json
import SimpleITK as sitk
from pathlib import Path
import numpy as np

def calculate_hd95(pred_img: sitk.Image, gt_img: sitk.Image) -> float:
    """
    Symmetric 95% Hausdorff Distance (HD95) between two binary segmentations.
    Uses physical spacing from the images => output unit is the image's physical unit (often mm).
    """
    # Ensure binary (0/1)
    pred = sitk.Cast(pred_img > 0, sitk.sitkUInt8)
    gt   = sitk.Cast(gt_img > 0, sitk.sitkUInt8)

    pred_sum = int(sitk.GetArrayViewFromImage(pred).sum())
    gt_sum   = int(sitk.GetArrayViewFromImage(gt).sum())

    # Handle empty cases
    if pred_sum == 0 and gt_sum == 0:
        return 0.0
    if pred_sum == 0 or gt_sum == 0:
        # undefined / very large; choose inf so you can spot failures
        return float("inf")

    # Surface distance A->B:
    # distance map of B, sampled on surface of A
    gt_dist = sitk.Abs(sitk.SignedMaurerDistanceMap(
        gt, insideIsPositive=False, squaredDistance=False, useImageSpacing=True
    ))
    pred_surf = sitk.LabelContour(pred)  # 1 on surface voxels
    d_pred_to_gt = sitk.GetArrayViewFromImage(gt_dist)[sitk.GetArrayViewFromImage(pred_surf) > 0]
    hd95_a = float(np.percentile(d_pred_to_gt, 95))

    # Surface distance B->A:
    pred_dist = sitk.Abs(sitk.SignedMaurerDistanceMap(
        pred, insideIsPositive=False, squaredDistance=False, useImageSpacing=True
    ))
    gt_surf = sitk.LabelContour(gt)
    d_gt_to_pred = sitk.GetArrayViewFromImage(pred_dist)[sitk.GetArrayViewFromImage(gt_surf) > 0]
    hd95_b = float(np.percentile(d_gt_to_pred, 95))

    return max(hd95_a, hd95_b)


def full_evaluation(SOURCE_DIR: Path, GT_DIR: Path, output: Path | None = None):
    predictions = {p.name for p in SOURCE_DIR.glob('*.nii.gz')}
    labels = {p.name for p in GT_DIR.glob('*.nii.gz')}
    if not predictions or predictions != labels:
        raise ValueError('Prediction and reference folders must contain the same nonempty case set')
    rows = []
    for name in sorted(predictions):
        pred_img, gt_img = sitk.ReadImage(str(SOURCE_DIR/name)), sitk.ReadImage(str(GT_DIR/name))
        if pred_img.GetSize() != gt_img.GetSize() or any(not np.allclose(a, b) for a, b in [
                (pred_img.GetSpacing(), gt_img.GetSpacing()), (pred_img.GetOrigin(), gt_img.GetOrigin()),
                (pred_img.GetDirection(), gt_img.GetDirection())]):
            raise ValueError(f'Prediction/reference grids differ: {name}')
        pred = sitk.GetArrayFromImage(pred_img) > 0
        gt = sitk.GetArrayFromImage(gt_img) > 0
        overlap, union = (pred & gt).sum(), (pred | gt).sum()
        # Match the original study definitions, including Dice=0 for empty references.
        dice = float(2*overlap/(pred.sum()+gt.sum())) if gt.any() else 0.
        rows.append(dict(case=name.removesuffix('.nii.gz'), dice=dice,
            iou=float(overlap/union) if union else 0., false_positive_voxels=int((pred & ~gt).sum()),
            hd95_mm=calculate_hd95(pred_img, gt_img)))
    dice = np.array([r['dice'] for r in rows])
    q1, median, q3 = np.quantile(dice, [.25, .5, .75])
    hd95 = [r['hd95_mm'] for r in rows if np.isfinite(r['hd95_mm'])]
    summary = dict(n=len(rows), mean_dice=float(dice.mean()), sd_dice=float(dice.std()),
        median_dice=float(median), dice_q25=float(q1), dice_q75=float(q3),
        LDR=float((dice > 0).mean()), mAP=float(np.mean([r['iou'] > .35355 for r in rows])),
        mean_false_positive_voxels=float(np.mean([r['false_positive_voxels'] for r in rows])),
        n_finite_hd95=len(hd95), mean_hd95_mm=float(np.mean(hd95)) if hd95 else None,
        median_hd95_mm=float(np.median(hd95)) if hd95 else None)
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        with (output/'cases.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute the recorded segmentation metrics.')
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='New directory for per-case CSV and summary JSON')
    args = parser.parse_args()
    full_evaluation(args.predictions, args.labels, args.output)
