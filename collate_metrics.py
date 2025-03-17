#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import random
from collections import defaultdict
from datetime import datetime
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    from tabulate import tabulate
    HAVE_TABULATE = True
except ImportError:
    HAVE_TABULATE = False

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def find_latest_pred_file(fold_dir, pred_file_name):
    """
    In the given fold directory, search all immediate subdirectories and return the path
    to the prediction file from the subdirectory with the latest modification time.
    """
    subdirs = [os.path.join(fold_dir, d) for d in os.listdir(fold_dir)
               if os.path.isdir(os.path.join(fold_dir, d))]
    valid = []
    for d in subdirs:
        candidate = os.path.join(d, pred_file_name)
        if os.path.isfile(candidate):
            mtime = os.path.getmtime(d)
            valid.append((candidate, mtime))
    if not valid:
        return None
    latest_file = max(valid, key=lambda x: x[1])[0]
    return latest_file

def compute_per_category_metrics(coco_gt, coco_dt, cat_id):
    """
    Compute segmentation metrics for a single category.
    Returns a list of 6 metrics: [AP, AP50, AP75, AP_s, AP_m, AP_l].
    """
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval.params.catIds = [cat_id]
    coco_eval.params.imgIds = list(coco_gt.getImgIds())
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[:6]

def compute_overall_metrics(coco_gt, coco_dt):
    """
    Compute overall segmentation metrics over all categories.
    Returns a list of 6 metrics.
    """
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval.params.imgIds = list(coco_gt.getImgIds())
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[:6]

def main():
    parser = argparse.ArgumentParser(
        description="For each fold (e.g. fold0, fold1, ...) in log_root, find the latest log folder (containing a prediction file),\n"
                    "compute COCO segmentation metrics (overall and per-category), and then print the mean and std of each metric across folds."
    )
    parser.add_argument("--log_root", type=str, default="output/endoscapes",
                        help="Root directory containing fold directories (e.g. fold0, fold1, ...).")
    parser.add_argument("--pred_file", type=str, default="test_preds.json",
                        help="Name of the COCO prediction file inside each log folder.")
    parser.add_argument("--gt_file", type=str, default="test_gt.json",
                        help="Path to the ground truth COCO JSON file.")
    args = parser.parse_args()

    # Find all fold directories in log_root (directories starting with "fold").
    fold_dirs = [os.path.join(args.log_root, d) for d in os.listdir(args.log_root)
                 if os.path.isdir(os.path.join(args.log_root, d)) and d.lower().startswith("fold")]
    if not fold_dirs:
        print("No fold directories found in log_root.")
        return

    overall_metrics_list = []
    per_cat_metrics = {}
    valid_folds = 0

    for fold_dir in fold_dirs:
        fold_name = os.path.basename(fold_dir)
        pred_path = find_latest_pred_file(fold_dir, args.pred_file)
        if pred_path is None:
            print(f"[{fold_name}] No prediction file '{args.pred_file}' found; skipping fold.")
            continue
        print(f"[{fold_name}] Using predictions from: {pred_path}")

        # For each fold, load the ground truth using the provided gt_file within the fold.
        gt_path = find_latest_pred_file(fold_dir, args.gt_file)
        if gt_path is None:
            print(f"[{fold_name}] No ground truth file '{args.gt_file}' found; skipping fold.")
            continue

        coco_gt = COCO(gt_path)
        cat_ids = coco_gt.getCatIds()
        cats = coco_gt.loadCats(cat_ids)
        cat_id_to_name = {cat["id"]: cat["name"] for cat in cats}
        for cat in cats:
            if cat["name"] not in per_cat_metrics:
                per_cat_metrics[cat["name"]] = []

        pred_data = load_json(pred_path)
        if "annotations" not in pred_data or len(pred_data["annotations"]) == 0:
            print(f"[{fold_name}] Prediction file '{args.pred_file}' has no annotations; skipping fold.")
            continue

        coco_dt = coco_gt.loadRes(pred_data["annotations"])
        overall = compute_overall_metrics(coco_gt, coco_dt)
        print(f"[{fold_name}] Overall metrics: {overall}")
        overall_metrics_list.append(overall)

        for cat_id in cat_ids:
            stats = compute_per_category_metrics(coco_gt, coco_dt, cat_id)
            per_cat_metrics[cat_id_to_name[cat_id]].append(stats)
        valid_folds += 1

    if valid_folds == 0:
        print("No valid folds processed. Exiting.")
        return

    overall_metrics_array = np.array(overall_metrics_list)
    overall_mean = np.mean(overall_metrics_array, axis=0)
    overall_std = np.std(overall_metrics_array, axis=0)

    print("\nOverall COCO segmentation metrics (averaged over folds):")
    print(f"mAP (0.5:0.95): {overall_mean[0]:.3f} ± {overall_std[0]:.3f}")
    print(f"AP50:           {overall_mean[1]:.3f} ± {overall_std[1]:.3f}")
    print(f"AP75:           {overall_mean[2]:.3f} ± {overall_std[2]:.3f}")
    print(f"AP_small:       {overall_mean[3]:.3f} ± {overall_std[3]:.3f}")
    print(f"AP_medium:      {overall_mean[4]:.3f} ± {overall_std[4]:.3f}")
    print(f"AP_large:       {overall_mean[5]:.3f} ± {overall_std[5]:.3f}")

    # Build per-category table.
    table = []
    for cat_name, metrics_list in per_cat_metrics.items():
        metrics_array = np.array(metrics_list)
        if metrics_array.size == 0:
            continue
        mean_metrics = np.mean(metrics_array, axis=0)
        std_metrics = np.std(metrics_array, axis=0)
        table.append([cat_name,
                      f"{mean_metrics[0]:.3f} ± {std_metrics[0]:.3f}",
                      f"{mean_metrics[1]:.3f} ± {std_metrics[1]:.3f}",
                      f"{mean_metrics[2]:.3f} ± {std_metrics[2]:.3f}",
                      f"{mean_metrics[3]:.3f} ± {std_metrics[3]:.3f}",
                      f"{mean_metrics[4]:.3f} ± {std_metrics[4]:.3f}",
                      f"{mean_metrics[5]:.3f} ± {std_metrics[5]:.3f}"])
    headers = ["Category", "mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
    print("\nPer-category COCO segmentation metrics (mean ± std over folds):")
    if HAVE_TABULATE:
        from tabulate import tabulate
        print(tabulate(table, headers=headers, tablefmt="grid"))
    else:
        header_line = "+----------------------+-----------+-----------+-----------+-----------+-----------+-----------+"
        print(header_line)
        print(f"| {headers[0]:<20} | {headers[1]:<9} | {headers[2]:<9} | {headers[3]:<9} | {headers[4]:<9} | {headers[5]:<9} | {headers[6]:<9} |")
        print(header_line)
        for row in table:
            print(f"| {row[0]:<20} | {row[1]:<9} | {row[2]:<9} | {row[3]:<9} | {row[4]:<9} | {row[5]:<9} | {row[6]:<9} |")
        print(header_line)

if __name__ == "__main__":
    main()
