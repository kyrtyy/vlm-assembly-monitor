"""
Vision Encoder
==============
Extracts spatial feature maps from RGB video frames.
Backbone: EfficientNet-B0 (chosen for edge compute awareness — ~5.3M params, 
0.39 GFLOPs at 224x224). Output is projected into a shared embedding space.

Architecture choice rationale:
- EfficientNet-B0 vs ViT: B0 has lower FLOPs/param ratio, ideal for TensorRT INT8.
- The final feature map (7x7x1280) is flattened to 49 visual tokens, each of 
  dimension d_model, matching the language token dimensionality for cross-attention.
"""
import torch
import torch.nn as nn
import timm
from einops import rearrange


class VisionEncoder(nn.Module):
    """
    Wraps EfficientNet-B0 (pretrained on ImageNet-21k via timm) and projects
    its spatial feature maps into d_model-dimensional visual tokens.

    For a single frame of shape (B, 3, H, W), outputs:
        visual_tokens : (B, num_patches, d_model)
        where num_patches = (H/32) * (W/32) for the EfficientNet stride-32 backbone.
        At H=W=224: num_patches = 7*7 = 49 tokens.
    """

    def __init__(
        self,
        d_model: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Load EfficientNet-B0, strip classifier, keep feature extractor
        # features_only=True gives us intermediate spatial maps
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            features_only=True,  # returns list of feature maps at each stage
            out_indices=(4,),    # stage 4 output: (B, 1280, H/32, W/32)
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # Channel count at stage 4 of EfficientNet-B0
        backbone_channels = 1280

        # Project backbone channels → d_model
        # 1x1 conv acts as a per-patch linear projection (identical to ViT patch embed)
        self.proj = nn.Sequential(
            nn.Conv2d(backbone_channels, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

        # Spatial positional embedding — learned, 7x7 = 49 positions by default.
        # We register as a parameter so it serialises with the model and resizes
        # gracefully in export_onnx.py if input resolution changes.
        self.pos_embed = nn.Parameter(torch.zeros(1, 49, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)  — normalised RGB frame(s).
        Returns:
            tokens: (B, N, d_model)  — N spatial visual tokens.
        """
        # Backbone: returns list, we asked for stage 4 only → (B, 1280, h, w)
        feats = self.backbone(x)[0]

        # 1x1 projection → (B, d_model, h, w)
        feats = self.proj(feats)

        # Flatten spatial dims → sequence of tokens: (B, d_model, N) → (B, N, d_model)
        B, C, h, w = feats.shape
        tokens = rearrange(feats, "b c h w -> b (h w) c")

        # Add positional embedding (resize bilinearly if spatial dims changed)
        if tokens.shape[1] != self.pos_embed.shape[1]:
            pos = self._resize_pos_embed(h, w)
        else:
            pos = self.pos_embed

        tokens = tokens + pos
        tokens = self.norm(self.dropout(tokens))
        return tokens

    def _resize_pos_embed(self, h: int, w: int) -> torch.Tensor:
        """Bicubic resize of positional embedding for non-standard input sizes."""
        hw0 = int(self.pos_embed.shape[1] ** 0.5)
        pos = self.pos_embed.reshape(1, hw0, hw0, self.d_model).permute(0, 3, 1, 2)
        pos = torch.nn.functional.interpolate(
            pos, size=(h, w), mode="bicubic", align_corners=False
        )
        return pos.permute(0, 2, 3, 1).reshape(1, h * w, self.d_model)


class TemporalVisionEncoder(nn.Module):
    """
    Applies VisionEncoder frame-by-frame over a video clip, then returns
    the per-frame token sequences stacked along a time dimension.

    For a clip of T frames: (B, T, 3, H, W) → (B, T, N, d_model)
    """

    def __init__(self, d_model: int = 512, pretrained: bool = True, **kwargs):
        super().__init__()
        self.encoder = VisionEncoder(d_model=d_model, pretrained=pretrained, **kwargs)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip: (B, T, 3, H, W)
        Returns:
            frame_tokens: (B, T, N, d_model)
        """
        B, T, C, H, W = clip.shape
        # Merge batch and time dimensions for a single encoder pass (efficient)
        frames = clip.view(B * T, C, H, W)
        tokens = self.encoder(frames)          # (B*T, N, d_model)
        N = tokens.shape[1]
        return tokens.view(B, T, N, self.encoder.d_model)
