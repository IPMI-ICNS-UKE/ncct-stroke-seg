import argparse
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


def full_evaluation(SOURCE_DIR: Path, GT_DIR: Path):
    print(f"Starting evaluation for predictions in {SOURCE_DIR} using ground truth in {GT_DIR}")

    dice_scores = []
    iou_scores = []
    FP_scores = []
    hd95_scores = []

    for prediction_file in SOURCE_DIR.iterdir():
        if prediction_file.suffix == ".gz":
            print(f"Found prediction {prediction_file}")

            gt_file = GT_DIR / prediction_file.name
            if not gt_file.exists():
                print(f"  -> No matching GT file found for {prediction_file.name} in {GT_DIR}, skipping.")
                continue

            # read prediction and ground truth
            pred_img = sitk.ReadImage(prediction_file)
            gt_img = sitk.ReadImage(gt_file)

            # convert to numpy arrays
            pred = sitk.GetArrayFromImage(pred_img)
            gt = sitk.GetArrayFromImage(gt_img)

            if pred.shape != gt.shape:
                print(f"  -> Shape mismatch for {prediction_file.name}: pred {pred.shape}, gt {gt.shape}, skipping.")
                continue

            # calculate metrics
            overlap = np.logical_and(pred, gt).sum()
            dice_score = 2 * overlap / (gt.sum() + pred.sum()) if gt.sum() else 0
            dice_scores.append(dice_score)

            union = np.logical_or(pred, gt).sum()
            iou_score = overlap / union if union > 0 else 0.0
            iou_scores.append(iou_score)

            false_positives = np.sum(~(gt.astype(bool)) & pred)
            FP_scores.append(false_positives)

            hd95 = calculate_hd95(pred_img, gt_img)
            hd95_scores.append(hd95)

            print(f"  -> DICE for {prediction_file.name}: {dice_score:.4f}")
            print(f"  -> IoU for {prediction_file.name}: {iou_score:.4f}")
            print(f"  -> HD95 for {prediction_file.name}: {hd95:.2f}")

    if len(dice_scores) == 0:
        print("No DICE scores were computed (no matching .gz files).")
        return None

    mean_dice = float(np.mean(dice_scores))
    sd = float(np.std(dice_scores))

    median_dice = float(np.median(dice_scores))
    q1 = np.percentile(dice_scores, 25)
    q3 = np.percentile(dice_scores, 75)

    #calculate mAP (mean Average Precision)
    threshold_map = 0.35355
    count_above = sum(1 for iou in iou_scores if iou > threshold_map)
    mAP = count_above / len(iou_scores)

    #calculate HD_accuracy
    count_above = sum(1 for dice in dice_scores if dice > 0)
    HD_accuracy = count_above / len(dice_scores)

    #mean fps
    mean_FP = float(np.mean(FP_scores))

    #valid hd95 values
    finite_hd95 = [h for h in hd95_scores if np.isfinite(h)]

    print(f"\nDICE computation finished.")
    print(f"Number of evaluated cases: {len(dice_scores)}")
    print(f"Mean DICE: {mean_dice:.4f} +- {sd:.4f}")
    print(f"Median DICE: {median_dice:.4f} [{q1:.4f};{q3:.4f}]")
    print(f"mAP: {mAP:.4f}")
    print(f"HD_accuracy: {HD_accuracy:.4f}")
    print(f"Mean FP: {mean_FP:.4f}")
    print(f"Mean HD95 (finite only): {float(np.mean(finite_hd95)) if finite_hd95 else float('nan'):.2f}")
    print(f"Median HD95 (finite only): {float(np.median(finite_hd95)) if finite_hd95 else float('nan'):.2f}")

    return mean_dice, median_dice, HD_accuracy, mean_FP, mAP

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the recorded segmentation metrics.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()
    full_evaluation(SOURCE_DIR=args.predictions, GT_DIR=args.labels)
