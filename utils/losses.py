"""
Loss Functions
==============
Combined multi-task loss for the VLM Assembly Monitor.

Total loss:
    L = λ_cls * L_CE  +  λ_box * L_GIoU  +  λ_obj * L_obj  +  λ_l1 * L_L1

Components:
    L_CE   — Cross-entropy with label smoothing for state classification.
    L_GIoU — Generalised IoU loss for bounding box regression.
    L_obj  — Binary cross-entropy for objectness (is there an object at slot i?).
    L_L1   — Smooth L1 regression loss (supplements GIoU for coordinate accuracy).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.heads import giou_loss


class AssemblyLoss(nn.Module):
    """
    Computes the combined multi-task loss for VLMAssemblyMonitor.

    Args:
        lambda_cls: Weight for state classification CE loss.
        lambda_box: Weight for GIoU bounding box loss.
        lambda_obj: Weight for objectness BCE loss.
        lambda_l1:  Weight for smooth L1 box regression loss.
        label_smoothing: Applied to CE loss (prevents over-confident states).
        num_states: Number of state classes.
    """

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_box: float = 2.0,
        lambda_obj: float = 0.5,
        lambda_l1:  float = 1.0,
        label_smoothing: float = 0.1,
        num_states: int = 6,
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_l1  = lambda_l1

        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction="mean",
        )

    def forward(
        self,
        predictions: dict,
        targets: dict,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            predictions: output dict from VLMAssemblyMonitor.forward()
                keys: 'state_logits', 'boxes', 'objectness'
            targets: ground-truth dict from DataLoader
                keys: 'state_label', 'boxes', 'box_mask'

        Returns:
            loss_dict with keys: 'total', 'cls', 'box', 'obj', 'l1'
        """
        state_logits = predictions["state_logits"]   # (B, num_states)
        pred_boxes   = predictions["boxes"]          # (B, T, K, 4)
        pred_obj     = predictions["objectness"]     # (B, T, K)

        gt_states    = targets["state_label"]        # (B,)
        gt_boxes     = targets["boxes"]              # (B, T, K, 4)
        box_mask     = targets["box_mask"]           # (B, T, K) bool

        # ── 1. State classification loss ──────────────────────────────────
        l_cls = self.ce_loss(state_logits, gt_states)

        # ── 2. Bounding box losses (only over valid gt boxes) ─────────────
        if box_mask.any():
            # Select valid positions
            pred_valid = pred_boxes[box_mask]   # (M, 4)
            gt_valid   = gt_boxes[box_mask]     # (M, 4)

            # GIoU loss
            l_box = giou_loss(pred_valid, gt_valid)

            # Smooth L1
            l_l1 = F.smooth_l1_loss(pred_valid, gt_valid, beta=0.1)
        else:
            l_box = pred_boxes.sum() * 0.0   # zero with grad
            l_l1  = pred_boxes.sum() * 0.0

        # ── 3. Objectness loss ────────────────────────────────────────────
        # box_mask is the "is there an object here?" ground truth
        l_obj = F.binary_cross_entropy_with_logits(
            pred_obj,
            box_mask.float(),
            reduction="mean",
        )

        # ── 4. Combine ────────────────────────────────────────────────────
        total = (
            self.lambda_cls * l_cls
            + self.lambda_box * l_box
            + self.lambda_obj * l_obj
            + self.lambda_l1  * l_l1
        )

        return {
            "total": total,
            "cls":   l_cls,
            "box":   l_box,
            "obj":   l_obj,
            "l1":    l_l1,
        }
