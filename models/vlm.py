"""
VLM Assembly Monitor — Full Pipeline
=====================================
Assembles all components into a single nn.Module:

    TemporalVisionEncoder   → (B, T, N, d_model)  [visual tokens per frame]
    LanguageEncoder         → (B, L, d_model)      [language tokens]
    CrossModalFusion        → (B, T, d_model)      [per-frame fused embeddings]
    CausalTemporalTransformer→ (B, T, d_model)     [temporal context]
    TemporalAggregator      → (B, d_model)         [clip embedding]
    StateClassifier         → (B, num_states)      [state logits]
    BBoxRegressor           → (B, T, K, 4) + (B, T, K) [boxes + objectness]

Forward pass overview:
    1. Each of T frames is encoded by TemporalVisionEncoder independently.
    2. The same instruction is encoded by LanguageEncoder once.
    3. CrossModalFusion runs per-frame: for each t ∈ [0,T), attend visual[t]
       with language, then mean-pool → scalar fused embedding (B, d_model).
    4. Stack T embeddings → (B, T, d_model), pass through CausalTransformer.
    5. TemporalAggregator pools to (B, d_model) for the classifier head.
    6. BBoxRegressor uses the full (B, T, d_model) temporal output for per-frame boxes.
"""
import torch
import torch.nn as nn

from models.vision_encoder import TemporalVisionEncoder
from models.language_encoder import LanguageEncoder
from models.fusion import CrossModalFusion
from models.temporal import CausalTemporalTransformer, TemporalAggregator
from models.heads import StateClassifier, BBoxRegressor


class VLMAssemblyMonitor(nn.Module):
    """
    Multimodal VLM for assembly state estimation and component tracking.

    Args:
        d_model:     Shared embedding dimension.  Default: 512.
        nhead:       Attention heads (must divide d_model).  Default: 8.
        num_states:  Number of discrete assembly states.     Default: 6.
        max_objects: Max tracked objects per frame.          Default: 4.
        T_max:       Maximum clip length (frames).           Default: 32.
        freeze_vision_backbone: Freeze EfficientNet weights (saves memory).
        freeze_bert:            Freeze DistilBERT weights.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_states: int = 6,
        max_objects: int = 4,
        T_max: int = 32,
        freeze_vision_backbone: bool = True,
        freeze_bert: bool = True,
    ):
        super().__init__()

        # ── Encoders ──────────────────────────────────────────────────────
        self.vision_enc = TemporalVisionEncoder(
            d_model=d_model,
            pretrained=True,
            freeze_backbone=freeze_vision_backbone,
        )
        self.lang_enc = LanguageEncoder(
            d_model=d_model,
            freeze_bert=freeze_bert,
        )

        # ── Fusion & temporal reasoning ───────────────────────────────────
        self.fusion = CrossModalFusion(
            d_model=d_model,
            nhead=nhead,
            ffn_dim=d_model * 4,
        )
        self.temporal = CausalTemporalTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=4,
            ffn_dim=d_model * 4,
            max_seq_len=T_max,
        )
        self.aggregator = TemporalAggregator(d_model=d_model)

        # ── Output heads ──────────────────────────────────────────────────
        self.state_head = StateClassifier(
            d_model=d_model,
            num_states=num_states,
        )
        self.bbox_head = BBoxRegressor(
            d_model=d_model,
            max_objects=max_objects,
        )

    def forward(
        self,
        clip: torch.Tensor,            # (B, T, 3, H, W)
        input_ids: torch.Tensor,       # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass.

        Returns a dict with keys:
            'state_logits'  : (B, num_states)
            'boxes'         : (B, T, max_objects, 4) — normalised (cx,cy,w,h)
            'objectness'    : (B, T, max_objects)    — objectness logits
            'attn_weights'  : (B, nhead, L, N)       — cross-attention maps (last frame)
        """
        B, T = clip.shape[:2]

        # 1. Encode all frames: (B, T, N, d_model)
        visual_tokens_seq = self.vision_enc(clip)

        # 2. Encode instruction once: (B, L, d_model), (B, L)
        lang_tokens, lang_mask = self.lang_enc(input_ids, attention_mask)

        # 3. Cross-modal fusion per frame (parallelized across T to avoid sequential bottlenecks)
        B, T, N, D = visual_tokens_seq.shape
        L = lang_tokens.shape[1]
        vis_flat = visual_tokens_seq.view(B * T, N, D)
        lang_flat = lang_tokens.repeat_interleave(T, dim=0)       # (B*T, L, d_model)
        lang_mask_flat = lang_mask.repeat_interleave(T, dim=0)   # (B*T, L)

        fused_flat, attn_w_flat = self.fusion(
            visual_tokens=vis_flat,
            lang_tokens=lang_flat,
            lang_padding_mask=lang_mask_flat,
        )                                                         # (B*T, L, d_model)
        
        # Mean-pool language dim → (B*T, d_model)
        pooled_flat = CrossModalFusion.mean_pool(fused_flat, lang_mask_flat)
        fused_seq = pooled_flat.view(B, T, D)                     # (B, T, d_model)

        # Extract attention weights for the last frame
        attn_w = attn_w_flat.view(B, T, self.fusion.nhead, L, N)
        attn_weights_last = attn_w[:, -1]                         # (B, nhead, L, N)

        # 4. Causal temporal transformer → (B, T, d_model)
        temporal_out = self.temporal(fused_seq)

        # 5. Aggregate → clip embedding → state logits
        clip_embed = self.aggregator(temporal_out)         # (B, d_model)
        state_logits = self.state_head(clip_embed)         # (B, num_states)

        # 6. Per-frame bounding boxes
        boxes, objectness = self.bbox_head(temporal_out)   # (B,T,K,4), (B,T,K)

        return {
            "state_logits": state_logits,
            "boxes": boxes,
            "objectness": objectness,
            "attn_weights": attn_weights_last,
        }

    def encode_instruction(
        self, instructions: list[str], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenise a list of instruction strings. Convenience wrapper."""
        enc = self.lang_enc.tokenize(instructions, device)
        return enc["input_ids"], enc["attention_mask"]

    def count_parameters(self) -> dict[str, int]:
        """Returns a breakdown of trainable parameter counts per sub-module."""
        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "vision_encoder": count(self.vision_enc),
            "language_encoder": count(self.lang_enc),
            "fusion": count(self.fusion),
            "temporal": count(self.temporal),
            "aggregator": count(self.aggregator),
            "state_head": count(self.state_head),
            "bbox_head": count(self.bbox_head),
            "total": count(self),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Import fix for fusion.CrossModalFusion.mean_pool (needs the class in scope)
# ──────────────────────────────────────────────────────────────────────────────
from models.fusion import CrossModalFusion  # noqa: F811 (re-import for clarity)
