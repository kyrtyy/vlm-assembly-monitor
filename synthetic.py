"""
Synthetic Dataset Generator
============================
Generates synthetic IKEA-style assembly clips for rapid prototyping on Colab
BEFORE you have access to the real IKEA ASM dataset.

Simulates:
- Video clips of T frames with coloured blocks representing furniture components
- Bounding boxes for each component (ground truth)
- Assembly state labels (0–5)
- Natural language instructions describing the assembly step

This lets you validate the full training pipeline end-to-end on any machine
without requiring dataset download or access.

Real dataset: https://ikeaasm.github.io/
"""
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T


# Assembly state taxonomy (matches StateClassifier num_states=6)
STATE_LABELS = [
    "not_started",
    "in_progress_early",
    "in_progress_mid",
    "in_progress_late",
    "completed",
    "error",
]

# Pool of synthetic assembly instructions
INSTRUCTIONS = [
    "Attach the wooden table leg to the central base constraint",
    "Insert the metal bolt through the side panel connector",
    "Align the shelf bracket with the left wall mounting point",
    "Secure the back panel using the four corner fasteners",
    "Connect the drawer rail to the lower cabinet frame",
    "Place the top surface onto the assembled leg assembly",
    "Tighten all visible screws using the provided hex key",
    "Slide the door panel into the upper and lower tracks",
]

# Synthetic component colours (simulates furniture parts)
COMPONENT_COLOURS = [
    (180, 120,  60),   # wood brown
    ( 80, 100, 130),   # metal blue
    (200, 200, 195),   # plastic grey
    (220, 170,  80),   # oak yellow
]


class SyntheticAssemblyDataset(Dataset):
    """
    Generates synthetic assembly clips on the fly.
    Each item is one clip of T frames with ground-truth state and bounding boxes.

    Args:
        num_clips:   Number of clips in the dataset.
        T:           Frames per clip.
        img_size:    (H, W) of each frame.
        num_objects: Number of assembly components per clip (1–4).
        max_objects: Max objects for BBoxRegressor (must match model config).
    """

    def __init__(
        self,
        num_clips: int = 1000,
        T: int = 8,
        img_size: tuple[int, int] = (224, 224),
        num_objects: int = 3,
        max_objects: int = 4,
        max_length: int = 64,
    ):
        self.num_clips = num_clips
        self.T = T
        self.img_size = img_size
        self.num_objects = min(num_objects, max_objects)
        self.max_objects = max_objects
        self.max_length = max_length

        # ImageNet normalisation (required by EfficientNet pretrained weights)
        self.frame_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # Pre-generate clip metadata (state + instruction per clip for reproducibility)
        rng = np.random.RandomState(42)
        self.clip_states = rng.randint(0, len(STATE_LABELS), size=num_clips)
        self.clip_instructions = [
            random.choice(INSTRUCTIONS) for _ in range(num_clips)
        ]

    def __len__(self) -> int:
        return self.num_clips

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            clip:         (T, 3, H, W)         float32 normalised frames
            instruction:  str                   natural language instruction
            state_label:  int                   assembly state index (0–5)
            boxes:        (T, max_objects, 4)   float32 gt (cx,cy,w,h) normalised
            box_mask:     (T, max_objects)       bool — True for valid boxes
        """
        rng = np.random.RandomState(idx)
        H, W = self.img_size
        state = int(self.clip_states[idx])
        instruction = self.clip_instructions[idx]

        # Generate object trajectories: each object moves linearly across frames
        # starting positions and velocities
        obj_start = rng.uniform(0.1, 0.7, size=(self.num_objects, 2))   # (cx0, cy0)
        obj_vel   = rng.uniform(-0.02, 0.02, size=(self.num_objects, 2)) # (vx, vy)
        obj_size  = rng.uniform(0.08, 0.20, size=(self.num_objects, 2))  # (w, h)
        obj_colours = [COMPONENT_COLOURS[i % len(COMPONENT_COLOURS)]
                       for i in range(self.num_objects)]

        frames = []
        boxes_seq = np.zeros((self.T, self.max_objects, 4), dtype=np.float32)
        box_mask  = np.zeros((self.T, self.max_objects), dtype=bool)

        for t in range(self.T):
            img = Image.new("RGB", (W, H), color=(240, 235, 225))
            draw = ImageDraw.Draw(img)

            # Simulate workbench background grid
            for gx in range(0, W, 40):
                draw.line([(gx, 0), (gx, H)], fill=(220, 215, 205), width=1)
            for gy in range(0, H, 40):
                draw.line([(0, gy), (W, gy)], fill=(220, 215, 205), width=1)

            for obj_i in range(self.num_objects):
                cx = float(np.clip(obj_start[obj_i, 0] + obj_vel[obj_i, 0] * t, 0.05, 0.95))
                cy = float(np.clip(obj_start[obj_i, 1] + obj_vel[obj_i, 1] * t, 0.05, 0.95))
                w  = float(obj_size[obj_i, 0])
                h  = float(obj_size[obj_i, 1])

                # Pixel coordinates
                x1 = int((cx - w / 2) * W)
                y1 = int((cy - h / 2) * H)
                x2 = int((cx + w / 2) * W)
                y2 = int((cy + h / 2) * H)

                colour = obj_colours[obj_i]
                draw.rectangle([x1, y1, x2, y2], fill=colour, outline=(50, 50, 50), width=2)

                # Draw object ID text for visual debugging
                draw.text((x1 + 4, y1 + 4), f"P{obj_i+1}", fill=(255, 255, 255))

                # Store normalised box
                boxes_seq[t, obj_i] = [cx, cy, w, h]
                box_mask[t, obj_i] = True

            # Simulate assembly progress: objects converge in later states
            if state >= 3:  # in_progress_late / completed
                convergence = rng.uniform(0.3, 0.8)
                draw.ellipse(
                    [int(W * 0.4), int(H * 0.4), int(W * 0.6), int(H * 0.6)],
                    outline=(0, 180, 0), width=3
                )

            # Random occlusion simulation: occasional black bar (steam/arm analogue)
            if rng.rand() < 0.15:
                occ_y = rng.randint(0, H - 20)
                draw.rectangle([0, occ_y, W, occ_y + 20], fill=(20, 20, 20))

            frames.append(self.frame_transform(img))

        clip = torch.stack(frames, dim=0)   # (T, 3, H, W)

        return {
            "clip":        clip,
            "instruction": instruction,
            "state_label": torch.tensor(state, dtype=torch.long),
            "boxes":       torch.tensor(boxes_seq),
            "box_mask":    torch.tensor(box_mask),
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate function that handles the variable instruction strings.
    PyTorch's default collate cannot stack strings; we keep them as a list.
    """
    return {
        "clip":        torch.stack([b["clip"] for b in batch]),
        "instruction": [b["instruction"] for b in batch],
        "state_label": torch.stack([b["state_label"] for b in batch]),
        "boxes":       torch.stack([b["boxes"] for b in batch]),
        "box_mask":    torch.stack([b["box_mask"] for b in batch]),
    }
