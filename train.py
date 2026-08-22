"""
Training Script — Multi-GPU DDP
=================================
Trains VLMAssemblyMonitor on synthetic or real IKEA ASM data.

Single-GPU:
    python train.py --synthetic --epochs 20 --batch_size 4

Multi-GPU (2x RTX 4090):
    torchrun --nproc_per_node=2 train.py --synthetic --epochs 30 --batch_size 8

Key features:
    - Distributed Data Parallel (DDP) via torchrun / NCCL backend
    - Mixed-precision (bf16 default on Ada Lovelace, fp16 fallback)
    - torch.compile() kernel optimization (--compile flag)
    - Gradient accumulation (enables larger effective batch sizes)
    - Cosine LR schedule with linear warmup
    - Enhanced WandB logging (gradients, per-step metrics, model artifacts)
    - Checkpoint saving / resuming (rank-0 only)
    - Early stopping on validation loss
"""
import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
import numpy as np

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from models.vlm import VLMAssemblyMonitor
from utils.losses import AssemblyLoss
from data.synthetic import SyntheticAssemblyDataset, collate_fn


# ── Distributed helpers ──────────────────────────────────────────────────────

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed():
    """Initialize the NCCL process group for multi-GPU training."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        return True
    return False


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


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
    p.add_argument("--compile",     action="store_true",
                   help="Apply torch.compile() for kernel optimization")
    # Training
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--grad_accum",  type=int, default=4,
                   help="Gradient accumulation steps (effective_bs = bs * grad_accum * num_gpus)")
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--warmup_steps",type=int, default=200)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--clip_grad",   type=float, default=1.0)
    p.add_argument("--val_split",   type=float, default=0.1)
    p.add_argument("--patience",    type=int, default=10,
                   help="Early stopping patience (epochs)")
    p.add_argument("--precision",   default="bf16", choices=["bf16", "fp16", "fp32"],
                   help="Mixed-precision dtype (bf16 recommended for RTX 4090)")
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

    # ── Distributed sampler (shards data across GPUs) ─────────────────────
    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed() else None
    val_sampler   = DistributedSampler(val_ds, shuffle=False) if is_distributed() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),  # shuffle only if not using DistributedSampler
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
    )
    return train_loader, val_loader, train_sampler


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine annealing to 1e-6."""
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(1e-6, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_amp_config(args, device):
    """
    Configure mixed-precision based on hardware capabilities.
    bf16 is preferred on Ada Lovelace (RTX 4090) — no GradScaler needed.
    fp16 requires GradScaler for loss scaling.
    """
    if args.precision == "fp32" or device.type == "cpu":
        return None, None, False

    if args.precision == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, None, True  # bf16 doesn't need scaler
        else:
            print("  bf16 not supported on this GPU, falling back to fp16.")
            args.precision = "fp16"

    # fp16 path
    scaler = GradScaler("cuda")
    return torch.float16, scaler, True


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_dtype) -> dict[str, float]:
    model.eval()
    total_losses = {}
    total_correct, total_samples = 0, 0

    # Unwrap DDP model for encode_instruction
    raw_model = model.module if isinstance(model, DDP) else model

    for batch in loader:
        clip         = batch["clip"].to(device, non_blocking=True)
        state_labels = batch["state_label"].to(device, non_blocking=True)
        gt_boxes     = batch["boxes"].to(device, non_blocking=True)
        box_mask     = batch["box_mask"].to(device, non_blocking=True)
        instructions = batch["instruction"]

        input_ids, attn_mask = raw_model.encode_instruction(instructions, device)

        ctx = autocast("cuda", dtype=amp_dtype) if amp_dtype else nullcontext()
        with ctx:
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
    # ── Distributed setup ─────────────────────────────────────────────────
    distributed = setup_distributed()
    local_rank = get_local_rank()
    rank = get_rank()
    world_size = get_world_size()

    set_seed(args.seed + rank)  # different seed per rank for data diversity

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        print(f"Device: {device} | World size: {world_size} | Precision: {args.precision}")
        if distributed:
            for i in range(world_size):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # ── WandB (rank 0 only) ──────────────────────────────────────────────
    if not args.no_wandb and is_main_process():
        try:
            import wandb
            wandb.init(
                project="vlm-assembly-monitor",
                config=vars(args),
                tags=[
                    "synthetic" if args.synthetic else "ikea_asm",
                    f"{world_size}xGPU",
                    args.precision,
                ],
                name=f"d{args.d_model}_bs{args.batch_size}x{world_size}_lr{args.lr}",
                save_code=True,
            )
        except Exception:
            print("WandB unavailable — logging to console only.")
            args.no_wandb = True
    elif not is_main_process():
        args.no_wandb = True  # silence non-rank-0 processes

    # ── Data ─────────────────────────────────────────────────────────────
    if is_main_process():
        print("Building dataloaders...")
    train_loader, val_loader, train_sampler = build_dataloaders(args)
    if is_main_process():
        print(f"  Train: {len(train_loader.dataset)} clips | Val: {len(val_loader.dataset)} clips")
        print(f"  Effective batch size: {args.batch_size} × {args.grad_accum} × {world_size} = "
              f"{args.batch_size * args.grad_accum * world_size}")

    # ── Model ────────────────────────────────────────────────────────────
    if is_main_process():
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

    if is_main_process():
        param_counts = model.count_parameters()
        print("Trainable parameters per module:")
        for k, v in param_counts.items():
            print(f"  {k:25s}: {v:>10,}")

    # ── Wrap in DDP ───────────────────────────────────────────────────────
    # DDP must wrap BEFORE torch.compile — compile traces through DDP's
    # forward(), enabling correct gradient bucketing and allreduce fusion.
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    # Access raw model for encode_instruction (DDP wraps the module)
    # For compiled models, .module still works since compile is transparent.
    raw_model = model.module if isinstance(model, DDP) else model

    # ── torch.compile (optional, applied after DDP) ───────────────────────
    if args.compile:
        if is_main_process():
            print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead")

    # ── WandB gradient/parameter logging ──────────────────────────────────
    if not args.no_wandb:
        import wandb
        wandb.watch(raw_model, log="all", log_freq=50)

    # ── Optimiser & LR schedule ───────────────────────────────────────────
    # Use different LR for fine-tuned encoder vs newly initialised layers
    encoder_params = (
        list(raw_model.vision_enc.encoder.proj.parameters()) +
        list(raw_model.lang_enc.proj.parameters())
    )
    other_params = [
        p for n, p in raw_model.named_parameters()
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
    criterion = AssemblyLoss(num_states=args.num_states).to(device)

    # ── Mixed-precision config ────────────────────────────────────────────
    amp_dtype, scaler, use_amp = get_amp_config(args, device)
    if is_main_process():
        print(f"  AMP: {'enabled' if use_amp else 'disabled'} | dtype: {amp_dtype} | "
              f"GradScaler: {'yes' if scaler else 'no'}")

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if is_main_process():
            print(f"Resumed from epoch {start_epoch}")

    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────
    from contextlib import nullcontext

    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = {}
        t0 = time.time()

        # Set epoch for DistributedSampler (ensures different shuffling each epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            clip         = batch["clip"].to(device, non_blocking=True)
            state_labels = batch["state_label"].to(device, non_blocking=True)
            gt_boxes     = batch["boxes"].to(device, non_blocking=True)
            box_mask     = batch["box_mask"].to(device, non_blocking=True)
            instructions = batch["instruction"]

            input_ids, attn_mask = raw_model.encode_instruction(instructions, device)

            ctx = autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
            with ctx:
                preds = model(clip, input_ids, attn_mask)
                loss_dict = criterion(
                    preds,
                    {"state_label": state_labels, "boxes": gt_boxes, "box_mask": box_mask}
                )
                loss = loss_dict["total"] / args.grad_accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # ── Per-step WandB logging ────────────────────────────────
                if not args.no_wandb and global_step % 10 == 0:
                    import wandb
                    wandb.log({
                        "step/total_loss": loss_dict["total"].item(),
                        "step/cls_loss": loss_dict["cls"].item(),
                        "step/box_loss": loss_dict["box"].item(),
                        "step/obj_loss": loss_dict["obj"].item(),
                        "step/l1_loss": loss_dict["l1"].item(),
                        "step/lr": scheduler.get_last_lr()[0],
                        "global_step": global_step,
                    })

            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

        # ── Validation ───────────────────────────────────────────────────
        val_metrics = evaluate(model, val_loader, criterion, device, amp_dtype)
        val_loss = val_metrics["total"]

        n_batches = len(train_loader)
        train_summary = {f"train/{k}": v / n_batches for k, v in epoch_losses.items()}
        val_summary   = {f"val/{k}":   v for k, v in val_metrics.items()}
        elapsed = time.time() - t0

        if is_main_process():
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

        # ── Checkpoint (rank 0 only) ─────────────────────────────────────
        if is_main_process():
            ckpt = {
                "epoch": epoch,
                "model": raw_model.state_dict(),
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

                # Save best checkpoint as WandB Artifact
                if not args.no_wandb:
                    artifact = wandb.Artifact(
                        "best-model", type="model",
                        description=f"Best val_loss={best_val_loss:.4f}",
                        metadata=vars(args),
                    )
                    artifact.add_file(os.path.join(args.save_dir, "best.pth"))
                    wandb.log_artifact(artifact)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                    break

        # Synchronize early stopping across all ranks
        if distributed:
            should_stop = torch.tensor([1 if patience_counter >= args.patience else 0],
                                       device=device)
            dist.broadcast(should_stop, src=0)
            if should_stop.item():
                break

    if is_main_process():
        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    if not args.no_wandb:
        import wandb
        wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    args = parse_args()
    train(args)
