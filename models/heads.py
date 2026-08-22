"""
Output Heads
============
Two task-specific prediction heads attached to the shared temporal embedding:

1. StateClassifier   — predicts the discrete assembly completion state.
   Loss: Cross-entropy (label smoothing=0.1 prevents overconfidence).

2. BBoxRegressor     — predicts (cx, cy, w, h) normalised bounding boxes for
   relevant components in each frame.
   Loss: Generalised Intersection over Union (GIoU) + L1 smooth regression.

Both heads operate independently on the shared representation, making the
architecture a multi-task learning setup. The combined loss is:
    L_total = λ_cls * L_ce  +  λ_box * (L_giou + λ_l1 * L_l1)

where λ values are set in train.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateClassifier(nn.Module):
    """
    2-layer MLP mapping the clip embedding → assembly state logits.

    Assembly states (defined by the IKEA ASM dataset taxonomy, simplified):
        0: not_started
        1: in_progress_early   (< 33% components placed)
        2: in_progress_mid     (33–66%)
        3: in_progress_late    (> 66%)
        4: completed
        5: error / misassembled

    Args:
        d_model:    Input embedding dimensionality.
        num_states: Number of discrete state classes (default: 6).
        hidden_dim: Intermediate MLP width.
        dropout:    Regularisation.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_states: int = 6,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_states),
        )
        # Initialise final layer with small weights (stabilises early training)
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, clip_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip_embed: (B, d_model)
        Returns:
            logits: (B, num_states) — raw scores (apply softmax for probabilities).
        """
        return self.head(clip_embed)


class BBoxRegressor(nn.Module):
    """
    Per-frame bounding box prediction head.

    Predicts up to max_objects bounding boxes per frame in (cx, cy, w, h) format,
    normalised to [0, 1] relative to image dimensions.

    Each box also has an objectness score (confidence that the box contains a
    relevant component), which is used as weights in the GIoU loss.

    Args:
        d_model:      Input frame embedding dimensionality.
        max_objects:  Maximum number of object instances per frame (default: 4).
                      IKEA ASM typically has 2–4 active components visible.
    """

    def __init__(
        self,
        d_model: int = 512,
        max_objects: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.max_objects = max_objects
        out_dim = max_objects * 5  # 4 bbox coords + 1 objectness per object

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, frame_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            frame_embeds: (B, T, d_model) — per-frame temporal representations.
        Returns:
            boxes:      (B, T, max_objects, 4) — predicted (cx, cy, w, h) in [0,1].
            objectness: (B, T, max_objects)    — predicted objectness logits.
        """
        B, T, D = frame_embeds.shape
        raw = self.head(frame_embeds)  # (B, T, max_objects * 5)

        raw = raw.view(B, T, self.max_objects, 5)
        # cx, cy, w, h → sigmoid to constrain to [0, 1]
        boxes = torch.sigmoid(raw[..., :4])
        objectness = raw[..., 4]       # logits (apply sigmoid for scores)
        return boxes, objectness


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) → (x1, y1, x2, y2). Input/output: (..., 4)."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def giou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Generalised Intersection over Union loss (Rezatofighi et al., 2019).
    
    GIoU addresses the limitation of standard IoU where gradient is zero for
    non-overlapping boxes. GIoU adds a penalty term based on the enclosing box:
        GIoU = IoU - (|C - (A ∪ B)| / |C|)
    where C is the smallest enclosing box of A and B.

    Loss = 1 - GIoU  ∈ [0, 2].

    Args:
        pred_boxes:   (..., 4) in (cx, cy, w, h) normalised format.
        target_boxes: (..., 4) in (cx, cy, w, h) normalised format.
        weights:      Optional per-box weights (e.g. objectness scores). (...,)
    Returns:
        Scalar mean GIoU loss.
    """
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    tgt_xyxy = box_cxcywh_to_xyxy(target_boxes)

    # Intersection
    inter_x1 = torch.max(pred_xyxy[..., 0], tgt_xyxy[..., 0])
    inter_y1 = torch.max(pred_xyxy[..., 1], tgt_xyxy[..., 1])
    inter_x2 = torch.min(pred_xyxy[..., 2], tgt_xyxy[..., 2])
    inter_y2 = torch.min(pred_xyxy[..., 3], tgt_xyxy[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union
    pred_area = (pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp(min=0) * \
                (pred_xyxy[..., 3] - pred_xyxy[..., 1]).clamp(min=0)
    tgt_area  = (tgt_xyxy[..., 2] - tgt_xyxy[..., 0]).clamp(min=0) * \
                (tgt_xyxy[..., 3] - tgt_xyxy[..., 1]).clamp(min=0)
    union_area = pred_area + tgt_area - inter_area + 1e-7

    iou = inter_area / union_area

    # Enclosing box
    enc_x1 = torch.min(pred_xyxy[..., 0], tgt_xyxy[..., 0])
    enc_y1 = torch.min(pred_xyxy[..., 1], tgt_xyxy[..., 1])
    enc_x2 = torch.max(pred_xyxy[..., 2], tgt_xyxy[..., 2])
    enc_y2 = torch.max(pred_xyxy[..., 3], tgt_xyxy[..., 3])
    enc_area = ((enc_x2 - enc_x1) * (enc_y2 - enc_y1)).clamp(min=1e-7)

    giou = iou - (enc_area - union_area) / enc_area
    loss = 1 - giou

    if weights is not None:
        loss = loss * weights
    return loss.mean()
