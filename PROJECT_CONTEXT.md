# Project Context: VLM Assembly Monitor

## System & Environment
- **Local Client:** macOS (Darwin arm64, SSH client / development environment)
- **Compute Cluster:** Remote Linux node (Accessed via SSH)
- **Accelerators:** 2x NVIDIA GeForce RTX 4090 (24GB VRAM each, Ada Lovelace architecture, 48GB total combined VRAM)
- **Target Workload:** Distributed multi-GPU training (`DistributedDataParallel` / PyTorch `torchrun` / Hugging Face `accelerate`)
- **Key Hardware Directives:**
  - Leverage `bf16` or `fp16` mixed-precision by default to saturate Ada Lovelace Tensor Cores and maximize batch sizes.
  - Enable `torch.compile()` kernel optimization where applicable.
  - Enable `pin_memory=True` on PyTorch DataLoaders (already configured in `train.py`).
  - Structure all multi-GPU scripts to prevent single-GPU bottlenecks (avoid raw `torch.nn.DataParallel`).
- **Active virtualenv/conda path:** [TBD - to be verified on remote node]
- **CUDA/PyTorch versions:** [TBD - to be verified once remote connection is established]

## Current Architecture & Goal
- **Model Type:** Multimodal Vision-Language Model (VLM) for sequential state estimation and component tracking.
  - Vision Encoder: EfficientNet-B0 + spatial positional embedding
  - Language Encoder: DistilBERT + projection layer
  - Fusion: Cross-modal cross-attention
  - Temporal: Causal Transformer + temporal aggregator
  - Output Heads: State classifier + BBox regressor (GIoU Loss + L1 Loss)
  - Post-processing: Kalman Filter tracking
- **Dataset Details:** IKEA Assembly (IKEA ASM) dataset or synthetic sequential assembly clips.
- **Current Objective:** Transition the development workspace to `/Users/macbook/.gemini/antigravity/scratch/vlm_assembly`, verify code layout, configure environment context, and prepare the training script (`train.py`) for multi-GPU training (DDP / `torchrun`), implementing mixed-precision and compile optimizations.

## Recent Changes & Decisions
- **DDP Refactor Complete:** `train.py` fully refactored for multi-GPU DDP via `torchrun` with NCCL backend, `DistributedSampler`, rank-gated I/O, and synchronized early stopping.
- **Modern Mixed-Precision:** Replaced deprecated `torch.cuda.amp` with `torch.amp`. bf16 is default for RTX 4090 (Ada Lovelace); fp16 fallback with `GradScaler`.
- **torch.compile:** Added `--compile` flag for kernel optimization (`reduce-overhead` mode).
- **Enhanced WandB:** `wandb.watch()` for gradient/parameter logging, per-step loss metrics, model artifact checkpointing on best val loss.
- **Parallelized Fusion:** Eliminated sequential Python loop in `models/vlm.py` — cross-modal fusion now runs in a single batched forward pass across all T frames.
- **ONNX Export Fixes:** Fixed NameError (`QuantFormat`), output names mismatch, and added actual FP16 conversion via `onnxconverter-common`.
- **Git Repository Initialized:** Local repo on `main` branch, ready for remote push.

## Known Blockers & Errors
- **SSH Credentials Pending:** Waiting for user to provide remote cluster SSH details to set up the push-to-deploy workflow and verify environment.

## Next Actions
- [ ] Obtain SSH details and set up bare repo + post-receive hook for push-to-deploy.
- [ ] `git push cluster main` to deploy code to the cluster.
- [ ] Create conda env on cluster, install dependencies, `wandb login`.
- [ ] Verify GPU setup (CUDA, 2x RTX 4090, bf16 support).
- [ ] Run single-GPU validation, then multi-GPU `torchrun` training.
- [ ] Push to GitHub for portfolio.

