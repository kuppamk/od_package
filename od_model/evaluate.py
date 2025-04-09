from collections import defaultdict

import torch
import numpy as np
from torchvision.ops import box_iou


class Evaluator:
    """Evaluates object detection results using IoU-based metrics and mAP."""

    def __init__(self, predictions: list, num_classes: int):
        """Initializes the evaluator.

        Args:
            predictions (list): List of prediction dictionaries with keys:
                'pred_boxes', 'pred_labels', 'pred_scores',
                'gt_boxes', 'gt_labels'.
            num_classes (int): Total number of classes (excluding background).
        """
        self.predictions = predictions
        self.num_classes = num_classes

    def compute_overall_metrics(self, iou_thresh: float = 0.5):
        """Computes overall precision, recall, and F1 score across all classes.

        Args:
            iou_thresh (float): IoU threshold to consider a prediction as a match.

        Returns:
            tuple: (precision, recall, F1 score)
        """
        TP, FP, FN = 0, 0, 0

        for pred in self.predictions:
            preds = pred["pred_boxes"]
            gts = pred["gt_boxes"]

            if len(preds) == 0 and len(gts) == 0:
                continue
            elif len(preds) == 0:
                FN += len(gts)
                continue
            elif len(gts) == 0:
                FP += len(preds)
                continue

            ious = box_iou(preds, gts)
            matched = (ious.max(dim=1).values > iou_thresh).sum().item()
            TP += matched
            FP += len(preds) - matched
            FN += len(gts) - matched

        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        return precision, recall, f1

    def compute_classwise_ap(self, iou_thresh: float = 0.5):
        """Computes class-wise average precision (AP) and mean AP (mAP).

        Args:
            iou_thresh (float): IoU threshold to consider a prediction as a match.

        Returns:
            tuple:
                ap_per_class (dict): Dictionary mapping class_id to AP.
                mAP (float): Mean average precision across all classes.
        """
        pred_by_class = defaultdict(list)
        gt_by_class = defaultdict(list)

        for pred in self.predictions:
            pred_boxes = pred["pred_boxes"]
            pred_labels = pred["pred_labels"]
            pred_scores = pred["pred_scores"]
            gt_boxes = pred["gt_boxes"]
            gt_labels = pred["gt_labels"]

            for c in range(1, self.num_classes + 1):
                cls_pred_mask = pred_labels == c
                cls_gt_mask = gt_labels == c

                cls_pred_boxes = pred_boxes[cls_pred_mask]
                cls_pred_scores = pred_scores[cls_pred_mask]
                cls_gt_boxes = gt_boxes[cls_gt_mask]

                matched = torch.zeros(len(cls_gt_boxes))
                tp = torch.zeros(len(cls_pred_boxes))
                fp = torch.zeros(len(cls_pred_boxes))

                for i, pbox in enumerate(cls_pred_boxes):
                    if len(cls_gt_boxes) == 0:
                        fp[i] = 1
                        continue

                    ious = box_iou(pbox.unsqueeze(0), cls_gt_boxes)[0]
                    max_iou, max_idx = ious.max(0)

                    if max_iou >= iou_thresh and matched[max_idx] == 0:
                        tp[i] = 1
                        matched[max_idx] = 1
                    else:
                        fp[i] = 1

                pred_by_class[c].extend(
                    zip(cls_pred_scores.tolist(), tp.tolist(), fp.tolist())
                )
                gt_by_class[c].append(len(cls_gt_boxes))

        ap_per_class = {}
        for c in range(1, self.num_classes + 1):
            if len(pred_by_class[c]) == 0:
                ap_per_class[c] = 0.0
                continue

            pred_by_class[c].sort(key=lambda x: -x[0])
            scores, tps, fps = zip(*pred_by_class[c])

            scores = np.array(scores)
            tps = np.array(tps)
            fps = np.array(fps)

            tps_cum = np.cumsum(tps)
            fps_cum = np.cumsum(fps)

            recalls = tps_cum / (sum(gt_by_class[c]) + 1e-6)
            precisions = tps_cum / (tps_cum + fps_cum + 1e-6)

            ap = 0.0
            for r in np.linspace(0, 1, 11):
                p = precisions[recalls >= r].max() if np.any(recalls >= r) else 0
                ap += p / 11.0

            ap_per_class[c] = round(ap, 4)

        mAP = round(np.mean(list(ap_per_class.values())), 4)
        return ap_per_class, mAP
