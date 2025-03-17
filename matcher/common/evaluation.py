r""" Evaluate mask prediction """
import torch
import torch.nn.functional as F
import numpy as np
from pycocotools import mask as maskUtils

class Evaluator:
    r""" Computes intersection and union between prediction and ground-truth,
    resizes predicted masks if needed, and accumulates predictions in a COCO-style results dictionary for mAP evaluation.
    It also enforces that each predicted image's id matches the ground-truth.
    """
    @classmethod
    def initialize(cls, dataset):
        cls.ignore_index = 255
        cls.gt_dets = dataset.test_json_data  # Ground-truth COCO dictionary

        # Build a mapping from GT image file_name to its id.
        cls.gt_image_map = {}
        cls.gt_size_map = {}
        for img in cls.gt_dets.get("images", []):
            cls.gt_image_map[img["file_name"]] = img["id"]
            cls.gt_size_map[img["file_name"]] = (img['height'], img['width'])

        # Initialize COCO-style results structure.
        cls.coco_results = {"images": [], "annotations": []}

        # image_map will map query_img_name to the corresponding GT id.
        cls.image_map = {}
        cls.annotation_counter = 1

    @classmethod
    def classify_prediction(cls, pred_mask, batch):
        gt_mask = batch.get('query_mask')
        query_img = batch.get('query_img')  # Tensor of shape [B, C, H, W]
        query_img_name = batch.get('query_name')  # Can be a list or a single string

        # Apply ignore_index as in PASCAL-5i masks.
        query_ignore_idx = batch.get('query_ignore_idx')
        if query_ignore_idx is not None:
            assert torch.logical_and(query_ignore_idx, gt_mask).sum() == 0
            query_ignore_idx *= cls.ignore_index
            gt_mask = gt_mask + query_ignore_idx
            pred_mask[gt_mask == cls.ignore_index] = cls.ignore_index

        # Compute intersection and union per sample.
        area_inter, area_pred, area_gt = [], [], []
        for _pred_mask, _gt_mask in zip(pred_mask, gt_mask):
            _inter = _pred_mask[_pred_mask == _gt_mask]
            if _inter.size(0) == 0:
                _area_inter = torch.tensor([0, 0], device=_pred_mask.device)
            else:
                _area_inter = torch.histc(_inter, bins=2, min=0, max=1)
            area_inter.append(_area_inter)
            area_pred.append(torch.histc(_pred_mask, bins=2, min=0, max=1))
            area_gt.append(torch.histc(_gt_mask, bins=2, min=0, max=1))
        area_inter = torch.stack(area_inter).t()
        area_pred = torch.stack(area_pred).t()
        area_gt = torch.stack(area_gt).t()
        area_union = area_pred + area_gt - area_inter

        # --- Accumulate COCO predictions ---
        batch_size = pred_mask.size(0)
        for i in range(batch_size):
            if isinstance(query_img_name, list):
                img_name = query_img_name[i]
            else:
                img_name = query_img_name

            # Enforce that the query image name exists in the GT mapping.
            if img_name not in cls.gt_image_map:
                raise ValueError(f"Query image name '{img_name}' not found in ground-truth images.")
            gt_img_id = cls.gt_image_map[img_name]
            gt_size = cls.gt_size_map[img_name]

            # Enforce consistency in our image_map.
            if img_name not in cls.image_map:
                # Use the ground-truth mask shape (which is assumed to be the original size)
                H, W = gt_size
                image_entry = {
                    "id": gt_img_id,
                    "file_name": img_name,
                    "width": W,
                    "height": H
                }
                cls.coco_results["images"].append(image_entry)
                cls.image_map[img_name] = gt_img_id

            # Determine category id.
            if "class_id" in batch:
                if isinstance(batch["class_id"], list):
                    cat_id = batch["class_id"][i]
                else:
                    cat_id = batch["class_id"]
            else:
                cat_id = 1

            # resize pred mask
            pred_mask = F.interpolate(pred_mask[i].view(1, 1, *pred_mask[i].shape),
                    gt_size, mode='nearest').squeeze()

            # Convert predicted mask to numpy (binary).
            pred_np = pred_mask.cpu().numpy().astype(np.uint8)
            rle = maskUtils.encode(np.asfortranarray(pred_np))
            if isinstance(rle['counts'], bytes):
                rle['counts'] = rle['counts'].decode("utf-8")
            bbox = maskUtils.toBbox(rle).tolist()
            area = float(maskUtils.area(rle))
            annotation = {
                "id": cls.annotation_counter,
                "image_id": gt_img_id,
                "category_id": int(cat_id),
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                "score": 1.0  # Dummy score.
            }
            cls.coco_results["annotations"].append(annotation)
            cls.annotation_counter += 1

        return area_inter, area_union
