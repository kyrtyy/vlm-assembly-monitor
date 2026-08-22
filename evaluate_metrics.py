import torch
import numpy as np
from tqdm import tqdm
from models.vlm import VLMAssemblyMonitor
from data.synthetic import SyntheticAssemblyDataset, STATE_LABELS
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
from torchvision.ops import box_iou

def xywh_to_xyxy(boxes):
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)"""
    x_c, y_c, w, h = boxes.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pth")
    parser.add_argument("--T", type=int, default=8, help="Sequence length")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    
    model = VLMAssemblyMonitor(T_max=args.T, use_jepa=True).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    print("Generating validation dataset (100 clips)...")
    dataset = SyntheticAssemblyDataset(num_clips=100, T=args.T)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    correct_states = 0
    total_states = 0
    total_iou = 0.0
    total_boxes = 0
    jepa_losses = []

    print("Evaluating metrics...")
    with torch.no_grad():
        for batch in tqdm(loader):
            clip = batch["clip"].to(device)
            labels = batch["state_label"].to(device)
            gt_boxes = batch["boxes"].to(device)
            box_mask = batch["box_mask"].to(device)
            instructions = batch["instruction"]

            input_ids, attn_mask = model.encode_instruction(instructions, device)

            # We pass model in train() mode momentarily JUST to get the jepa_target out for evaluation
            model.train() 
            preds = model(clip, input_ids, attn_mask)
            model.eval()

            # 1. State Accuracy
            state_logits = preds["state_logits"]
            pred_classes = state_logits.argmax(dim=-1)
            correct_states += (pred_classes == labels).sum().item()
            total_states += labels.size(0)

            # 2. Bounding Box IoU
            pred_boxes = preds["boxes"]
            if box_mask.any():
                p_boxes = pred_boxes[box_mask] # (M, 4)
                g_boxes = gt_boxes[box_mask]   # (M, 4)
                
                p_xyxy = xywh_to_xyxy(p_boxes)
                g_xyxy = xywh_to_xyxy(g_boxes)
                
                # Diagonal IoU (approximate pairwise)
                iou = torch.diag(box_iou(p_xyxy, g_xyxy))
                total_iou += iou.sum().item()
                total_boxes += iou.size(0)

            # 3. JEPA Loss
            if "jepa_pred" in preds:
                j_pred = preds["jepa_pred"][:, :-1]
                j_tgt = preds["jepa_target"][:, 1:]
                j_loss = F.smooth_l1_loss(j_pred, j_tgt)
                jepa_losses.append(j_loss.item())

    acc = (correct_states / total_states) * 100
    mean_iou = (total_iou / total_boxes) * 100 if total_boxes > 0 else 0
    mean_jepa = np.mean(jepa_losses) if jepa_losses else 0

    print("\n" + "="*50)
    print(" 🚀 VLM TELEOPERATION MONITOR - QUANTITATIVE METRICS")
    print("="*50)
    print(f" State Classification Accuracy : {acc:.2f} %")
    print(f" Bounding Box Tracking (mIoU)  : {mean_iou:.2f} %")
    print(f" JEPA Predictive Latent Loss   : {mean_jepa:.4f} (Smooth L1)")
    print("="*50)
    print("\nPaste these numbers back to me, and I will finalize your README!")

if __name__ == "__main__":
    main()
