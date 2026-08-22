"""
Visual Inference Script
=======================
Runs the VLM Assembly Monitor on a video (or synthetic clip) and outputs
a visualised video with bounding boxes, state predictions, and cross-attention
overlays (optional).

Usage:
  python inference.py --checkpoint ./checkpoints/best.pth --synthetic --output demo.mp4
  python inference.py --checkpoint ./checkpoints/best.pth --video input.mp4 --instruction "Attach table leg"
"""
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from models.vlm import VLMAssemblyMonitor
from data.synthetic import SyntheticAssemblyDataset, STATE_LABELS
import torchvision.transforms as transforms

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pth")
    parser.add_argument("--video", type=str, default=None, help="Path to input MP4")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic moving-box video")
    parser.add_argument("--instruction", type=str, default="Assemble the wooden table", help="Language instruction")
    parser.add_argument("--output", type=str, default="output_demo.mp4", help="Output video path")
    parser.add_argument("--T", type=int, default=32, help="Sequence length")
    parser.add_argument("--img_size", type=int, default=224, help="Frame resolution")
    return parser.parse_args()

def denormalize(tensor):
    """Denormalize ImageNet tensor to numpy 0-255 image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor.cpu() * std + mean
    tensor = tensor.clamp(0, 1)
    img = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    # RGB to BGR for OpenCV
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def load_video_frames(video_path, max_frames=32, img_size=224):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (img_size, img_size))
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    # Pad if too short
    while len(frames) < max_frames:
        frames.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
        
    # Transform to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_frames = torch.stack([transform(f) for f in frames])
    return tensor_frames

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    # 1. Load Model
    print(f"Loading model from {args.checkpoint}...")
    model = VLMAssemblyMonitor(
        d_model=512, nhead=8, num_states=6, max_objects=4, T_max=args.T,
        freeze_vision_backbone=False, freeze_bert=False
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model = model.to(device)
    model.eval()

    # 2. Get Input Data
    print("Preparing input data...")
    if args.synthetic:
        dataset = SyntheticAssemblyDataset(num_clips=1, T=args.T, img_size=(args.img_size, args.img_size))
        item = dataset[0]
        clip = item["clip"].unsqueeze(0).to(device)
        instruction = item["instruction"]
    else:
        if not args.video:
            print("Error: Must specify --video or --synthetic")
            return
        clip = load_video_frames(args.video, args.T, args.img_size).unsqueeze(0).to(device)
        instruction = args.instruction

    print(f"Instruction: '{instruction}'")
    
    # Tokenize instruction
    input_ids, attn_mask = model.encode_instruction([instruction], device)

    # 3. Inference
    print("Running model inference...")
    with torch.no_grad():
        preds = model(clip, input_ids, attn_mask)
        
    state_logits = preds["state_logits"][0]
    boxes = preds["boxes"][0]        # (T, max_objects, 4)
    objectness = preds["objectness"][0] # (T, max_objects)
    
    predicted_state_idx = state_logits.argmax().item()
    predicted_state_label = STATE_LABELS[predicted_state_idx]
    
    # 4. Visualization
    print("Generating visualised video...")
    clip_cpu = clip[0].cpu() # (T, 3, H, W)
    boxes_cpu = boxes.cpu().numpy()
    obj_cpu = objectness.sigmoid().cpu().numpy()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, 5.0, (args.img_size, args.img_size))
    
    for t in range(args.T):
        frame = denormalize(clip_cpu[t])
        
        # Draw State & Instruction
        cv2.putText(frame, f"State: {predicted_state_label}", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Cmd: {instruction[:30]}...", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Draw Boxes
        for k in range(boxes_cpu.shape[1]):
            prob = obj_cpu[t, k]
            if prob > 0.5: # Objectness threshold
                cx, cy, w, h = boxes_cpu[t, k]
                # Denormalize coordinates
                x1 = int((cx - w/2) * args.img_size)
                y1 = int((cy - h/2) * args.img_size)
                x2 = int((cx + w/2) * args.img_size)
                y2 = int((cy + h/2) * args.img_size)
                
                color = (0, int(255*prob), 255) # Yellow-ish
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{prob:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
        out.write(frame)
        
    out.release()
    print(f"Done! Visualisation saved to {args.output}")

if __name__ == "__main__":
    main()
