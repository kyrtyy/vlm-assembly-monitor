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
- **Workspace Shift:** Shifted work from fluid super-resolution to the `vlm_assembly` project per user directive.
- **Implementation Plan Approved:** The implementation plan for refactoring `train.py` to support PyTorch DDP / `torchrun`, mixed-precision, and model compilation was reviewed and approved.
- **Codebase Review and Analysis:** Conducted a thorough codebase review ([codebase_review.md](file:///Users/macbook/.gemini/antigravity/brain/d758ea84-afc3-45eb-adf5-22f2d405ad33/codebase_review.md)), identifying key bugs in `export_onnx.py` (NameError, output names mismatch, lack of actual FP16 conversion) and an optimization target in `models/vlm.py` (sequential loop over frames in cross-modal fusion).
- **Task List Initialized:** Created [task.md](file:///Users/macbook/.gemini/antigravity/brain/d758ea84-afc3-45eb-adf5-22f2d405ad33/task.md) to manage execution.

## Known Blockers & Errors
- **SSH Credentials Pending:** Waiting for the user to provide remote node SSH details to connect, inspect the environment, and run validation.

## Next Actions
- [ ] Implement the parallelized cross-modal fusion optimization in `models/vlm.py`.
- [ ] Fix the ONNX export bugs (NameError, output names mismatch, actual FP16 conversion) in `export_onnx.py`.
- [ ] Refactor `train.py` to support PyTorch DDP (`torchrun`), mixed-precision (`bf16`), and `torch.compile()`.
- [ ] Connect to the remote node via SSH (once credentials provided) to verify CUDA, PyTorch, Conda environment, and path.
- [ ] Verify refactored DDP training script on remote node.
