"""
Training Script
===============
Trains VLMAssemblyMonitor on synthetic or real IKEA ASM data.

Colab usage:
    !python train.py --synthetic --epochs 20 --batch_size 4

Real dataset:
    !python train.py --data_root /content/ikea_asm_dataset_public --epochs 50

Key features:
    - Mixed-precision (FP16) training via torch.amp (cuts memory ~40%)
    - Gradient accumulation (enables larger effective batch sizes on T4 GPU)
    - Cosine LR schedule with linear warmup
    - WandB logging (set --no_wandb to disable)
    - Checkpoint saving / resuming
    - Early stopping on validation loss
"""
import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import numpy as np

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from models.vlm import VLMAssemblyMonitor
from utils.losses import AssemblyLoss
from data.synthetic import SyntheticAssemblyDataset, collate_fn


def parse_args():
    p = argparse.ArgumentParser(description="Train VLM Assembly Monitor")
    # Data
    p.add_argument("--synthetic",   action="store_true",
                   help="Use synthetic dataset (no IKEA ASM download required)")
    p.add_argument("--data_root",   default="./ikea_asm",
                   help="Path to ikea_asm_dataset_public/")
    p.add_argument("--num_clips",   type=int, default=2000,
                   help="Number of synthetic clips to generate")
    p.add_argument("--T",           type=int, default=8,
                   help="Frames per clip")
    p.add_argument("--img_size",    type=int, default=224,
                   help="Frame resolution (square)")
    p.add_argument("--num_objects", type=int, default=3)
    # Model
    p.add_argument("--d_model",     type=int, default=512)
    p.add_argument("--nhead",       type=int, default=8)
    p.add_argument("--num_states",  type=int, default=6)
    p.add_argument("--max_objects", type=int, default=4)
    p.add_argument("--freeze_vis",  action="store_true", default=True,
                   help="Freeze EfficientNet backbone (recommended for Colab)")
    p.add_argument("--freeze_bert", action="store_true", default=True,
                   help="Freeze DistilBERT (recommended for Colab)")
    # Training
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--grad_accum",  type=int, default=4,
                   help="Gradient accumulation steps (effective_bs = bs * grad_accum)")
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--warmup_steps",type=int, default=200)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--clip_grad",   type=float, default=1.0)
    p.add_argument("--val_split",   type=float, default=0.1)
    p.add_argument("--patience",    type=int, default=10,
                   help="Early stopping patience (epochs)")
    # Misc
    p.add_argument("--save_dir",    default="./checkpoints")
    p.add_argument("--resume",      default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--no_wandb",    action="store_true")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(args) -> tuple[DataLoader, DataLoader]:
    if args.synthetic:
        full_dataset = SyntheticAssemblyDataset(
            num_clips=args.num_clips,
            T=args.T,
            img_size=(args.img_size, args.img_size),
            num_objects=args.num_objects,
            max_objects=args.max_objects,
        )
    else:
        from data.dataset import IKEAAsmDataset
        full_dataset = IKEAAsmDataset(
            root=args.data_root,
            split="train",
            T=args.T,
            img_size=(args.img_size, args.img_size),
            max_objects=args.max_objects,
            augment=True,
        )

    val_size  = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine annealing to 1e-6."""
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(1e-6, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict[str, float]:
    model.eval()
    total_losses = {}
    total_correct, total_samples = 0, 0

    for batch in loader:
        clip         = batch["clip"].to(device)
        state_labels = batch["state_label"].to(device)
        gt_boxes     = batch["boxes"].to(device)
        box_mask     = batch["box_mask"].to(device)
        instructions = batch["instruction"]

        input_ids, attn_mask = model.encode_instruction(instructions, device)

        with autocast():
            preds = model(clip, input_ids, attn_mask)
            loss_dict = criterion(
                preds,
                {"state_label": state_labels, "boxes": gt_boxes, "box_mask": box_mask}
            )

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()

        # State accuracy
        pred_states = preds["state_logits"].argmax(dim=1)
        total_correct  += (pred_states == state_labels).sum().item()
        total_samples  += state_labels.size(0)

    n = len(loader)
    metrics = {k: v / n for k, v in total_losses.items()}
    metrics["state_accuracy"] = total_correct / max(total_samples, 1)
    return metrics


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── WandB ────────────────────────────────────────────────────────────
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(
                project="vlm-assembly-monitor",
                config=vars(args),
                tags=["synthetic" if args.synthetic else "ikea_asm"],
            )
        except Exception:
            print("WandB unavailable — logging to console only.")
            args.no_wandb = True

    # ── Data ─────────────────────────────────────────────────────────────
    print("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(args)
    print(f"  Train: {len(train_loader.dataset)} clips | Val: {len(val_loader.dataset)} clips")

    # ── Model ────────────────────────────────────────────────────────────
    print("Building model...")
    model = VLMAssemblyMonitor(
        d_model=args.d_model,
        nhead=args.nhead,
        num_states=args.num_states,
        max_objects=args.max_objects,
        T_max=args.T,
        freeze_vision_backbone=args.freeze_vis,
        freeze_bert=args.freeze_bert,
    ).to(device)

    param_counts = model.count_parameters()
    print("Trainable parameters per module:")
    for k, v in param_counts.items():
        print(f"  {k:25s}: {v:>10,}")

    # ── Optimiser & LR schedule ───────────────────────────────────────────
    # Use different LR for fine-tuned encoder vs newly initialised layers
    encoder_params = (
        list(model.vision_enc.encoder.proj.parameters()) +
        list(model.lang_enc.proj.parameters())
    )
    other_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not any(
            p is ep for ep in encoder_params
        )
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr * 0.1},  # lower LR for pretrained
            {"params": other_params,   "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(train_loader) // args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    scaler = GradScaler()
    criterion = AssemblyLoss(num_states=args.num_states).to(device)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────
    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = {}
        t0 = time.time()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            clip         = batch["clip"].to(device, non_blocking=True)
            state_labels = batch["state_label"].to(device, non_blocking=True)
            gt_boxes     = batch["boxes"].to(device, non_blocking=True)
            box_mask     = batch["box_mask"].to(device, non_blocking=True)
            instructions = batch["instruction"]

            input_ids, attn_mask = model.encode_instruction(instructions, device)

            with autocast():
                preds = model(clip, input_ids, attn_mask)
                loss_dict = criterion(
                    preds,
                    {"state_label": state_labels, "boxes": gt_boxes, "box_mask": box_mask}
                )
                loss = loss_dict["total"] / args.grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

        # ── Validation ───────────────────────────────────────────────────
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics["total"]

        n_batches = len(train_loader)
        train_summary = {f"train/{k}": v / n_batches for k, v in epoch_losses.items()}
        val_summary   = {f"val/{k}":   v for k, v in val_metrics.items()}
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"train_loss={epoch_losses['total']/n_batches:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['state_accuracy']*100:.1f}% | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.1f}s"
        )

        if not args.no_wandb:
            import wandb
            wandb.log({**train_summary, **val_summary, "epoch": epoch + 1})

        # ── Checkpoint ──────────────────────────────────────────────────
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.save_dir, "last.pth"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(ckpt, os.path.join(args.save_dir, "best.pth"))
            print(f"  → New best val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
