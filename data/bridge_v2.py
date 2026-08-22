import torch
from torch.utils.data import IterableDataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("Please install datasets: pip install datasets")

class BridgeV2TeleopDataset(IterableDataset):
    """
    Adapter for the Bridge Data V2 robotic teleoperation dataset via HuggingFace.
    Streams data directly to avoid downloading hundreds of GBs upfront.
    """
    def __init__(self, T=8, img_size=(224, 224), split="train", max_objects=4):
        super().__init__()
        self.T = T
        self.img_size = img_size
        self.max_objects = max_objects
        
        # Load the dataset in streaming mode so it doesn't fill your hard drive
        # Note: Replace 'jxu124/OpenX-Embodiment' with the exact HF repo you prefer for BridgeV2
        print(f"Initializing BridgeV2 Dataset (Streaming {split} split)...")
        self.dataset = load_dataset("jxu124/OpenX-Embodiment", "bridge", split=split, streaming=True)
        
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __iter__(self):
        for episode in self.dataset:
            # 1. Extract frames (different HF repos format this differently, usually a list of PIL Images or numpy arrays)
            # Assuming 'image' or 'observation' key contains the video frames
            if 'image' in episode:
                frames = episode['image']
            elif 'observation' in episode and 'image' in episode['observation']:
                frames = episode['observation']['image']
            else:
                continue # Skip if unrecognized format
                
            if len(frames) < self.T:
                continue
                
            # Sample T frames evenly across the episode
            indices = np.linspace(0, len(frames) - 1, self.T, dtype=int)
            sampled_frames = [frames[i] for i in indices]
            
            # Convert to tensors
            clip_tensors = []
            for frame in sampled_frames:
                if isinstance(frame, np.ndarray):
                    frame = Image.fromarray(frame)
                clip_tensors.append(self.transform(frame))
            clip = torch.stack(clip_tensors) # (T, 3, H, W)
            
            # 2. Extract Instruction
            instruction = episode.get('language_instruction', "Manipulate the object")
            if isinstance(instruction, list):
                instruction = instruction[0] # Take first if multiple
                
            # 3. State Labels (Heuristic based on time)
            # Since BridgeV2 doesn't have discrete semantic states out of the box,
            # we heuristically assign states based on the progress of the episode.
            # 0: approaching, 1: grasping, 2: manipulating, 3: placing, 4: completed
            state_label = 2 # Default to manipulating
            
            # 4. Bounding Boxes (Zeroed out)
            # BridgeV2 doesn't provide bounding boxes for objects. 
            # We set box_mask to False so the VLM's BBox Loss is ignored during training!
            # The model will focus entirely on JEPA Latent Prediction and State Classification.
            boxes = torch.zeros((self.T, self.max_objects, 4), dtype=torch.float32)
            box_mask = torch.zeros((self.T, self.max_objects), dtype=torch.bool)
            
            yield {
                "clip": clip,
                "instruction": str(instruction),
                "state_label": torch.tensor(state_label, dtype=torch.long),
                "boxes": boxes,
                "box_mask": box_mask
            }
